# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference,create_queries_and_maps_from_dataset
import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def model_structure(model):
    from thop import profile
    flops,params= profile(model,inputs=(torch.randn(1,3,800,800).cuda(),None,['person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'],))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # model_structure(model)
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(data_loader.dataset,
                                                                                        cfg)
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        # wyj: resize the image input!!!!!!!!!!!!!!!!!
        # toim = images.tensors.numpy()
        # toshow = toim[0, :, :, :]
        # toshow = toshow.transpose((1, 2, 0))
        # import matplotlib.pyplot as plt
        # plt.imshow(toshow)
        # plt.show()
        # from skimage import transform
        # import numpy as np
        # toshow=transform.resize(toshow,(800,800,3))
        # toshow = toshow.transpose((2, 0, 1))
        # toshow=np.expand_dims(toshow,axis=0)
        # images.tensors=torch.from_numpy(toshow)
        # # images.image_sizes=[torch.Size([800,800])]

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            with autocast():
                if len(captions) > 0:
                    if cfg.print_flops:
                        from thop import profile
                        flops, params = profile(model, inputs=(images, targets, captions, positive_map,greenlight_map))
                        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                        print('Params = ' + str(params / 1000 ** 2) + 'M')
                    loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
                    USE_CAM = False
                    if USE_CAM:
                        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
                            XGradCAM, EigenCAM, FullGrad
                        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                        from pytorch_grad_cam.utils.image import show_cam_on_image
                        target_layers = [model.backbone.body.layers[-1].blocks[-1].norm1]
                        input_tensor = images.tensors  # Create an input tensor image for your model..
                        # Note: input_tensor can be a batch tensor with several images!

                        # Construct the CAM object once, and then re-use it on many images:
                        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                        targetsori = [ClassifierOutputTarget(281)]

                        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                        grayscale_cam = cam(input_tensor=input_tensor,targets=targetsori, targets0=targets,captions=captions, positive_map=positive_map, greenlight_map = greenlight_map)

                        # In this example grayscale_cam has only one image in the batch:
                        grayscale_cam = grayscale_cam[0, :]
                        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        enter_val_phase=False
        try:
            if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
                enter_val_phase=True
        except:
            if checkpoint_period==0:
                enter_val_phase=True
        if enter_val_phase:
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False
                    )
                    if is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    if cfg.USE_CAM and i>300:
                        continue
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images,captions=all_queries,positive_map=all_positive_map_label_to_token[0])#wuyongjian:caution,remember to change prompt!!!!!!!!!!!!!!
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        if cfg.USE_CAM:
                            import warnings
                            warnings.filterwarnings('ignore')
                            warnings.simplefilter('ignore')
                            coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
                                          'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
                                          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                                          'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
                                          'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                                          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                          'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                                          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                                          'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                          'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
                                          'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                          'microwave',
                                          'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
                                          'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                            import cv2
                            from pytorch_grad_cam import AblationCAM, EigenCAM, ScoreCAM
                            from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
                            from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
                            from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
                            from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                scale_accross_batch_and_channels, scale_cam_image
                            target_layers = [model.backbone]
                            pred_labels=output[0].extra_fields['labels'].detach().cpu().numpy()
                            pred_scores = output[0].extra_fields['scores'].detach().cpu().numpy()
                            pred_classes = [coco_names[i] for i in pred_labels]
                            pred_boxes=output[0].bbox.detach().cpu().numpy()
                            # boxes, classes, labels, indices = [], [], [], []
                            # for index in range(len(pred_scores)):
                            #     if pred_scores[index] >= detection_threshold:
                            #         boxes.append(pred_bboxes[index].astype(np.int32))
                            #         classes.append(pred_classes[index])
                            #         labels.append(pred_labels[index])
                            #         indices.append(index)
                            # boxes = np.int32(boxes)
                            # return boxes, classes, labels, indices
                            targets = [FasterRCNNBoxScoreTarget(labels=pred_labels, bounding_boxes=pred_boxes)]
                            input_tensor = images.tensors
                            def draw_boxes(boxes, labels, classes, image):
                                COLORS=[[255,0,255],[255,0,0],[0,0,255]]
                                for i, box in enumerate(boxes):
                                    color =[1,0,0]# COLORS[labels[i]]
                                    cv2.rectangle(
                                        image,
                                        (int(box[0]), int(box[1])),
                                        (int(box[2]), int(box[3])),
                                        color, 2
                                    )
                                    # cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                                    #             lineType=cv2.LINE_AA)
                                return image

                            def fasterrcnn_reshape_transform(x):
                                target_size = x[0].size()[-2:]
                                activations = []
                                for value in x[0:1]:
                                    activations.append(
                                        torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
                                activations = torch.cat(activations, axis=1)
                                return activations

                            cam = EigenCAM(model,
                                           target_layers,
                                           use_cuda=torch.cuda.is_available(),
                                           reshape_transform=fasterrcnn_reshape_transform)
                            grayscale_cam = cam(input_tensor, targets=targets,targets0=None,
                                  captions=all_queries, positive_map=all_positive_map_label_to_token[0])

                            # import matplotlib.pyplot as plt
                            # plt.imshow(IMG[:,:,(2,1,0)])
                            #
                            # Take the first image in the batch:
                            grayscale_cam = grayscale_cam[0, :]

                            # plt.subplot(2, 2, 1)
                            id_to_img=val_data_loader.dataset.id_to_img_map
                            IMG_path=val_data_loader.dataset.root+'/'+val_data_loader.dataset.coco.imgs[id_to_img[image_ids[0]]]['file_name']
                            from skimage import io,transform
                            IMG=transform.resize(io.imread(IMG_path),grayscale_cam.shape)
                            # plt.imshow(IMG)
                            IMGori=IMG
                            try:
                                cam_image = show_cam_on_image(IMGori, grayscale_cam, use_rgb=True)
                            except:
                                continue
                            from skimage import io,transform
                            savpath='PLOT2/'+cfg.MODEL.WEIGHT.replace('/','_').replace('.','_')
                            if not os.path.exists(savpath):
                                os.mkdir(savpath)
                            io.imsave(savpath+'/'+val_data_loader.dataset.coco.imgs[id_to_img[image_ids[0]]]['file_name'],cam_image)
                            ORIsavpath = 'ORI_WITH_BOX'
                            if not os.path.exists(ORIsavpath):
                                os.mkdir(ORIsavpath)
                            ORIIM_PATH=ORIsavpath+'/'+val_data_loader.dataset.coco.imgs[id_to_img[image_ids[0]]]['file_name']
                            if not os.path.exists(ORIIM_PATH):
                                img = io.imread(IMG_path)
                                dataset = val_data_loader.dataset.coco.dataset
                                image_id = image_ids[0]
                                if img is None:
                                    continue
                                # img = cv2.cvtColor(imori, cv2.COLOR_BGR2RGB)
                                boxes = [ann['bbox'] for ann in dataset['annotations'] if ann['image_id'] == id_to_img[image_id]]
                                try:
                                    scores = [ann['score'] for ann in dataset['annotations'] if
                                              ann['image_id'] == id_to_img[image_id]]
                                except:
                                    scores = [1 for ann in dataset['annotations'] if ann['image_id'] == id_to_img[image_id]]
                                cls_ids = [1 for ann in dataset['annotations'] if ann['image_id'] == id_to_img[image_id]]
                                from maskrcnn_benchmark.engine.visualize import vis
                                this_image_vis = vis(img, boxes, scores, cls_ids, conf=0.3, class_names=None,
                                                     col=[255, 0, 0])
                                # plt.imshow(this_image_vis)
                                # plt.show()
                                io.imsave(ORIIM_PATH, this_image_vis)
                            # image_with_bounding_boxes = draw_boxes(gtboxes, pred_labels, pred_classes, IMG)
                            # io.imsave(ORIsavpath + '/' + val_data_loader.dataset.coco.imgs[image_ids[0]]['file_name'],
                            #           image_with_bounding_boxes)

                            # # And lets draw the boxes again:
                            # image_with_bounding_boxes = draw_boxes(pred_boxes, pred_labels, pred_classes, cam_image)

                            # plt.subplot(2, 2, 2)
                            # plt.imshow(cam_image)
                            # plt.subplot(2, 2, 3)
                            # plt.imshow(grayscale_cam)
                            # plt.show()
                            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                            # grayscale_cam = cam(input_tensor=input_tensor, targets=targetsori, targets0=None,
                            #                     captions=all_queries, positive_map=all_positive_map_label_to_token[0])
                            #
                            # # In this example grayscale_cam has only one image in the batch:
                            # grayscale_cam = grayscale_cam[0, :]
                            # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC,cfg=cfg)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result
                    checkpointer.save("model_best", **arguments)
                print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                    break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
