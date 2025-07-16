# -- coding: utf-8 --**
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import os
import json
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
# cocogt_dataset = json.load(f)
# newims = []
# x = os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
# x.sort()
# for image_id, name in enumerate(x):
#     newims.append({"id": int(name[:-4]),
#                    "height": 250, "width": 250, "file_name": name})
# cocogt_dataset['images'] = newims
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
# json.dump(cocogt_dataset, f)
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train(cfg, local_rank, distributed, use_tensorboard=False,):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False
    
    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    if cfg.USE_TRAIN_COPY:
        from maskrcnn_benchmark.engine.trainer_copy import do_train
    else:
        from maskrcnn_benchmark.engine.trainer import do_train
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters
    )

    return model

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def preprocess_raw_glip_result(jsonfile='LAST_PREDICT_BBOXS.json',visual=False):#wuyongjian edited : convert bboxs.json to a pseudo label jsonfile,which will be feed into the next cycle to fine-tune a new glip
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("DATASET/coco/annotations/instances_train2017_glipGT.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("DATASET/coco/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    # if visual:
    #     from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
    #     savdir = 'val_{}'.format(jsonfile).replace('.', '_')
    #     try:
    #         os.mkdir(savdir)
    #         vis_dataset(cocodt_dataset, savdir)
    #     except:
    #         print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def change_yolox_label_to_glip_label(jsonfile='instances_train_0193.json',dataset_num=''):
    try:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)['annotations']
    except:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)
    f = open("DATASET/coco{}/annotations/instances_train2017_glipGT.json".format(dataset_num), 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = cocodt_dataset_ann
    f = open("DATASET/coco{}/annotations/instances_train2017.json".format(dataset_num), 'w')
    json.dump(cocodt_dataset, f)
    # import time
    # print('sleeping........')
    # time.sleep(10)
    # f2017r = open("DATASET/coco/annotations/instances_train2017.json", 'r')
    # data2017 = json.load(f2017r)
    # return
def change_yolox_labelS_to_glip_labelS(labels=['instances_train_0193.json',]):
    ann = []
    for labeljson in labels:
        f = open(labeljson, 'r')
        cur_ann = json.load(f)
        ori_len = len(ann)
        for box in cur_ann:
            box['id'] = box['id'] + ori_len
            ann.append(box)
    f = open("DATASET/coco/annotations/instances_train2017_glipGT.json", 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = ann
    f2017 = open("DATASET/coco/annotations/instances_train2017.json", 'w')
    json.dump(cocodt_dataset, f2017)
def main():
    # import time
    # time.sleep(3600*10)
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--restart", default=False)
    parser.add_argument("--train_label", default=None)#"DATASET/coco/annotations/instances_train2017_glipGT.json")
    parser.add_argument("--dataset_num", default="")
    parser.add_argument("--train_labels",default=None,nargs='+')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.train_labels:
        print('converge train labels into a single label...................................................................')
        labels=args.train_labels
        change_yolox_labelS_to_glip_labelS(labels)#wuyongjian: it is wired .if you don't write this as a function, JSON libiary will always failed to write the train.json file, like missing some lines.
    elif args.train_label:
        change_yolox_label_to_glip_label(args.train_label,dataset_num=args.dataset_num)
    if args.distributed:
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )
    
    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.SWINBLO = 0
    cfg.lang_adap_mlp = 0
    cfg.vl_cross_att = 0
    cfg.fuse_module_cross_att=0
    cfg.generalized_vl = 0
    # cfg.LOCATION = 'pad'
    # cfg.defrost()
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    if args.restart:
        import shutil
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))
    
    save_config(cfg, output_config_path)

    model = train(cfg=cfg,
                  local_rank=args.local_rank,
                  distributed=args.distributed,
                  use_tensorboard=args.use_tensorboard)
from pycocotools.coco import COCO
import shutil
import numpy as np
from skimage import measure,io,transform
import matplotlib.pyplot as plt
def prepare_for_CONSEP_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco2/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1*=(250/256)
            y1*=(250/256)
            x2*=(250/256)
            y2*=(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco2/annotations/refined_instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CONSEP_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco22/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 2, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 3, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 4, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 5, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 6, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 7, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco22/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CCRCC_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco33/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            try:
                cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            except:
                cls=0
                print('wired!!!!!!!!!!!')
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 4, "supercategory": 'endothelial nuclei', "name": 'endothelial nuclei'},
                                  {"id": 1, "supercategory": 'tumor nuclei with grade 1', "name": 'tumor nuclei with grade 1'},
                                  {"id": 2, "supercategory": 'tumor nuclei with grade 2', "name": 'tumor nuclei with grade 2'},
                                  {"id": 3, "supercategory": 'tumor nuclei with grade 3', "name": 'tumor nuclei with grade 3'},
                                  ]
    # pass
    with open('DATASET/coco33/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box*(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection2(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        # shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        img=io.imread('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME))
        output=transform.resize(img,(250,250))
        io.imsave(savp+'%012d.jpg'%(image_id),output)
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1*=(250/256)
            y1*=(250/256)
            x2*=(250/256)
            y2*=(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)

# if __name__ == "__main__":
#     # prepare_for_CONSEP_GT_detection(phase='Train')
#     # prepare_for_CONSEP_GT_detection(phase='Val')
#     # prepare_for_CONSEP_multiclass_GT_detection(phase='Train')
#     # prepare_for_CONSEP_multiclass_GT_detection(phase='Val')
#     # prepare_for_CCRCC_multiclass_GT_detection(phase='Train')
#     # prepare_for_CCRCC_multiclass_GT_detection(phase='Val')
#     # prepare_for_CCRCC_GT_detection2(phase='Train')
#     # prepare_for_CCRCC_GT_detection2(phase='Val')
#     # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-10 19:17:37.741144.json")
#     # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-11 16:34:15.150666.json")
#     # import time
#     # time.sleep(2000)
#
#     # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set0.json",'r')
#     # cocogt_dataset = json.load(f,strict=False)
#     # annos=cocogt_dataset['annotations']
#     # images=cocogt_dataset['images']
#     # for im in images:
#     #     im['file_name']=im['file_name'].replace('COCO_train2014_','')
#     # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set1.json",'w')
#     # json.dump(cocogt_dataset,f)
#
#     # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json", 'r')
#     # cocogt_dataset = json.load(f)
#     # annos=cocogt_dataset['annotations']
#     # for anno in annos:
#     #     anno.update({'iscrowd': 0})
#     # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name_iscrowd.json", 'w')
#     # json.dump(cocogt_dataset,f)
#
#     # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
#     # cocogt_dataset = json.load(f)
#     # newims=[]
#     # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
#     # x.sort()
#     # for image_id,name in enumerate(x):
#     #     newims.append({"id": int(name[:-4]),
#     #                                    "height": 250, "width": 250, "file_name": name})
#     # cocogt_dataset['images']=newims
#     # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
#     # json.dump(cocogt_dataset,f)
#     main()
# -- coding: utf-8 --**
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import os
import json
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
# cocogt_dataset = json.load(f)
# newims = []
# x = os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
# x.sort()
# for image_id, name in enumerate(x):
#     newims.append({"id": int(name[:-4]),
#                    "height": 250, "width": 250, "file_name": name})
# cocogt_dataset['images'] = newims
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
# json.dump(cocogt_dataset, f)
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train(cfg, local_rank, distributed, use_tensorboard=False,):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False

    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    if cfg.USE_TRAIN_COPY:
        from maskrcnn_benchmark.engine.trainer_copy import do_train
    else:
        from maskrcnn_benchmark.engine.trainer import do_train
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters
    )

    return model

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def preprocess_raw_glip_result(jsonfile='LAST_PREDICT_BBOXS.json',visual=False):#wuyongjian edited : convert bboxs.json to a pseudo label jsonfile,which will be feed into the next cycle to fine-tune a new glip
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("DATASET/coco/annotations/instances_train2017_glipGT.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("DATASET/coco/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    # if visual:
    #     from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
    #     savdir = 'val_{}'.format(jsonfile).replace('.', '_')
    #     try:
    #         os.mkdir(savdir)
    #         vis_dataset(cocodt_dataset, savdir)
    #     except:
    #         print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def change_yolox_label_to_glip_label(jsonfile='instances_train_0193.json',dataset_num=''):
    try:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)['annotations']
    except:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)
    f = open("DATASET/coco{}/annotations/instances_train2017_glipGT.json".format(dataset_num), 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = cocodt_dataset_ann
    f = open("DATASET/coco{}/annotations/instances_train2017.json".format(dataset_num), 'w')
    json.dump(cocodt_dataset, f)
    # import time
    # print('sleeping........')
    # time.sleep(10)
    # f2017r = open("DATASET/coco/annotations/instances_train2017.json", 'r')
    # data2017 = json.load(f2017r)
    # return
def change_yolox_labelS_to_glip_labelS(labels=['instances_train_0193.json',]):
    ann = []
    for labeljson in labels:
        f = open(labeljson, 'r')
        cur_ann = json.load(f)
        ori_len = len(ann)
        for box in cur_ann:
            box['id'] = box['id'] + ori_len
            ann.append(box)
    f = open("DATASET/coco/annotations/instances_train2017_glipGT.json", 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = ann
    f2017 = open("DATASET/coco/annotations/instances_train2017.json", 'w')
    json.dump(cocodt_dataset, f2017)
def main():
    # import time
    # time.sleep(3600*10)
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--restart", default=False)
    parser.add_argument("--train_label", default=None)#"DATASET/coco/annotations/instances_train2017_glipGT.json")
    parser.add_argument("--dataset_num", default="")
    parser.add_argument("--train_labels",default=None,nargs='+')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.train_labels:
        print('converge train labels into a single label...................................................................')
        labels=args.train_labels
        change_yolox_labelS_to_glip_labelS(labels)#wuyongjian: it is wired .if you don't write this as a function, JSON libiary will always failed to write the train.json file, like missing some lines.
    elif args.train_label:
        change_yolox_label_to_glip_label(args.train_label,dataset_num=args.dataset_num)
    if args.distributed:
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )

    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    # cfg.LOCATION = 'pad'
    # cfg.defrost()
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    if args.restart:
        import shutil
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))

    save_config(cfg, output_config_path)

    model = train(cfg=cfg,
                  local_rank=args.local_rank,
                  distributed=args.distributed,
                  use_tensorboard=args.use_tensorboard)
from pycocotools.coco import COCO
import shutil
import numpy as np
from skimage import measure,io,transform
import matplotlib.pyplot as plt
def prepare_for_CONSEP_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco2/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1 -= 2
            y1 -= 2
            x2 += 2
            y2 += 2
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco2/annotations/refined_instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CONSEP_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco22/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 2, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 3, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 4, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 5, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 6, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 7, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco22/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CCRCC_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco33/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            try:
                cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            except:
                cls=0
                print('wired!!!!!!!!!!!')
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 4, "supercategory": 'endothelial nuclei', "name": 'endothelial nuclei'},
                                  {"id": 1, "supercategory": 'tumor nuclei with grade 1', "name": 'tumor nuclei with grade 1'},
                                  {"id": 2, "supercategory": 'tumor nuclei with grade 2', "name": 'tumor nuclei with grade 2'},
                                  {"id": 3, "supercategory": 'tumor nuclei with grade 3', "name": 'tumor nuclei with grade 3'},
                                  ]
    # pass
    with open('DATASET/coco33/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box*(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection2(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        # shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        img=io.imread('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME))
        output=transform.resize(img,(250,250))
        io.imsave(savp+'%012d.jpg'%(image_id),output)
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1*=(250/256)
            y1*=(250/256)
            x2*=(250/256)
            y2*=(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)

if __name__ == "__main__":
    # prepare_for_CONSEP_GT_detection(phase='Train')
    # prepare_for_CONSEP_GT_detection(phase='Val')
    # prepare_for_CONSEP_multiclass_GT_detection(phase='Train')
    # prepare_for_CONSEP_multiclass_GT_detection(phase='Val')
    # prepare_for_CCRCC_multiclass_GT_detection(phase='Train')
    # prepare_for_CCRCC_multiclass_GT_detection(phase='Val')
    # prepare_for_CCRCC_GT_detection2(phase='Train')
    # prepare_for_CCRCC_GT_detection2(phase='Val')
    # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-10 19:17:37.741144.json")
    # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-11 16:34:15.150666.json")
    # import time
    # time.sleep(2000)

    # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set0.json",'r')
    # cocogt_dataset = json.load(f,strict=False)
    # annos=cocogt_dataset['annotations']
    # images=cocogt_dataset['images']
    # for im in images:
    #     im['file_name']=im['file_name'].replace('COCO_train2014_','')
    # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set1.json",'w')
    # json.dump(cocogt_dataset,f)

    # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json", 'r')
    # cocogt_dataset = json.load(f)
    # annos=cocogt_dataset['annotations']
    # for anno in annos:
    #     anno.update({'iscrowd': 0})
    # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name_iscrowd.json", 'w')
    # json.dump(cocogt_dataset,f)

    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
    # cocogt_dataset = json.load(f)
    # newims=[]
    # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
    # x.sort()
    # for image_id,name in enumerate(x):
    #     newims.append({"id": int(name[:-4]),
    #                                    "height": 250, "width": 250, "file_name": name})
    # cocogt_dataset['images']=newims
    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
    # json.dump(cocogt_dataset,f)
    # f = open("DATASET/odinw/PascalVOC/train/annotations_without_background.json", 'r')
    # cocogt_dataset = json.load(f)
    # newims=[]
    # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
    # x.sort()
    # for image_id,name in enumerate(x):
    #     newims.append({"id": int(name[:-4]),
    #                                    "height": 250, "width": 250, "file_name": name})
    # cocogt_dataset['images']=newims
    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
    # json.dump(cocogt_dataset,f)
    main()
