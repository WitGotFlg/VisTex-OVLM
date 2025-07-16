import logging
import tempfile
import os

import torch
import numpy as np
import json

from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
# from maskrcnn_benchmark.utils.visualize import *
from maskrcnn_benchmark.utils.visualize import vis_dataset,draw_3color_bboxes_on_images,draw_2color_bboxes_on_images,compute_metrics,compute_metrics2
from skimage import measure, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def do_coco_evaluation(
        dataset,
        predictions,
        box_only,
        output_folder,
        iou_types,
        expected_results,
        expected_results_sigma_tol,cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        if dataset.coco is None and output_folder:
            json_results = prepare_for_tsv_detection(predictions, dataset)
            with open(os.path.join(output_folder, "box_proposals.json"), "w") as f:
                json.dump(json_results, f)
            return None
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return res, predictions
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        if dataset.coco is None:
            coco_results["bbox"] = prepare_for_tsv_detection(predictions, dataset)
        else:
            coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset, cfg=cfg)
    if "segm" in iou_types and cfg.MODEL.MASK_ON:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset)

    results = COCOResults(*iou_types)
    if 'coco0_grounding_train' in cfg.DATASETS.TRAIN or 'lvis_val' in cfg.DATASETS.TEST or 'odinw_val' in cfg.DATASETS.TEST or cfg.SOLVER.TEST_WITH_INFERENCE:
        for iou_type in iou_types:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_folder:
                    file_path = os.path.join(output_folder, iou_type + ".json")
                if dataset.coco:
                    res = evaluate_predictions_on_coco(
                        dataset.coco, coco_results[iou_type], file_path, iou_type
                    )
                    if cfg.plot_tsne:
                        import copy
                        orires = copy.deepcopy(res)
                        RELABEL=np.zeros(len(res.cocoDt.anns))
                        for ann_id,ann_key in enumerate(res.cocoDt.anns):
                            the_ann=res.cocoDt.anns[ann_key]
                            the_ann['category_id']=1
                        for ann_id,ann_key in enumerate(res.cocoGt.anns):
                            the_ann = res.cocoGt.anns[ann_key]
                            the_ann['category_id']=1
                        res.evaluate()
                        res.accumulate()
                        res.summarize()
                        # for ann_id,ann_key in enumerate(res.cocoDt.anns):
                        #     the_ann=res.cocoDt.anns[ann_key]
                        image_ids = res.params.imgIds
                        for iter, image_id in enumerate(image_ids):
                            cat_id=0
                            this_image_pred_ious = res.ious[(image_id, 1)]#the_ann['category_id'] has been 1
                            if len(this_image_pred_ious)==0:
                                continue
                            for resorted_ann_id in range(this_image_pred_ious.shape[0]):
                                this_box_iou=this_image_pred_ious[resorted_ann_id,:]
                                possible_resorted_gtbox_id=np.argmax(this_box_iou)
                                possible_resorted_dtbox_id=resorted_ann_id
                                possible_original_gtbox_id=res.evalImgs[cat_id * 4 * len(image_ids) + iter]['gtIds'][possible_resorted_gtbox_id]
                                possible_original_dtbox_id = res.evalImgs[cat_id * 4 * len(image_ids) + iter]['dtIds'][possible_resorted_dtbox_id]
                                original_gtbox=orires.cocoGt.anns[possible_original_gtbox_id]
                                original_gtbox_label=original_gtbox['category_id']
                                RELABEL[possible_original_dtbox_id]=original_gtbox_label
                        np.save('/home/data/jy/GLIP/RELABEL.npy',RELABEL)
                        res=copy.deepcopy(orires)

                        all_array = np.load('/home/data/jy/GLIP/consep_feature_base.npz')
                        data = all_array['arr_0']
                        label = all_array['arr_1']

                        CENTER_POINTS = np.zeros((int(max(RELABEL)), 256))
                        for i in range(int(max(RELABEL))):
                            first_targetcat_id = np.where(RELABEL == i + 1)[0][0]
                            CENTER_POINTS[i, :] = data[first_targetcat_id, :]
                        from sklearn.metrics.pairwise import euclidean_distances
                        distances = euclidean_distances(data, CENTER_POINTS)
                        labels = np.argmin(distances, axis=1)+1
                        # from sklearn.cluster import KMeans
                        # kmeans = KMeans(n_clusters=7, random_state=42,init=CENTER_POINTS,n_int=1)
                        # kmeans.fit(data)
                        # labels = kmeans.labels_
                        # label60 = labels[:60]
                        # centers = kmeans.cluster_centers_
                        # print("Cluster centers:")
                        # print(centers)
                        # SWITCH={2:-4,0:-5,6:-5,1:-6,3:-4,5:-4}
                        labelGT_KMEAN = np.concatenate(([label, ], [labels, ],[RELABEL,]), axis=0)
                        # SWITCH_labels=copy.deepcopy(labels)
                        #
                        # for id,key in enumerate(SWITCH):
                        #     print('switch {} to {}'.format(key,SWITCH[key]))
                        #     SWITCH_labels[SWITCH_labels==key]=SWITCH[key]
                        for ann_id,ann_key in enumerate(res.cocoDt.anns):
                            the_ann=res.cocoDt.anns[ann_key]
                            if RELABEL[ann_id]!=label[ann_id]:
                                the_ann['category_id']=labels[ann_id]
                            else:
                                the_ann['category_id'] = label[ann_id]
                        res.evaluate()
                        res.accumulate()
                        res.summarize()
                    results.update(res)
                elif output_folder:
                    with open(file_path, "w") as f:
                        json.dump(coco_results[iou_type], f)

        logger.info(results)
        check_expected_results(results, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(results, os.path.join(output_folder, "coco_results.pth"))
        if cfg.VISUALIZE :#and ('coco2_2017_val' in cfg.DATASETS.TEST or 'coco1_2017_val' in cfg.DATASETS.TEST) :#wuyongjian: plot for paper,no more needed
            cuter = results.results['bbox']['AP']
            try:
                coconumber=cfg.DATASETS.TEST[0][:]
            except:
                coconumber='___'
            if not os.path.exists('OUTPUT{}'.format(coconumber)):
                os.mkdir('OUTPUT{}'.format(coconumber))
            savepath='OUTPUT{}'.format(coconumber)+'/%.5f'%cuter
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            # cocoEval = COCOeval(res.cocoGt, res.cocoDt, 'bbox')
            # cocoEval.evaluate()
            # cocoEval.accumulate()
            if cfg.IMPROMPT.gvl==-1:
                draw_2color_bboxes_on_images(res, savepath, valdata_dir=dataset.root,THRE=cfg.THRE,dataset=dataset,NEED_LABELBOX=cfg.NEED_LABELBOX,count_green_ap=cfg.green_counter,mask_on=(cfg.MODEL.MASK_ON and cfg.MASK_ON_WHEN_TEST),GT_MASK_ON=cfg.GT_MASK_ON,mask2contour=cfg.mask2contour)
            else:
                draw_2color_bboxes_on_images(res, savepath, valdata_dir=dataset.root,THRE=cfg.THRE, dataset=dataset,
                                             NEED_LABELBOX=cfg.NEED_LABELBOX,count_green_ap=cfg.green_counter,mask_on=cfg.MODEL.MASK_ON and cfg.MASK_ON_WHEN_TEST,GT_MASK_ON=cfg.GT_MASK_ON,mask2contour=cfg.mask2contour)
            if cfg.MODEL.MASK_ON and cfg.mask2contour:
                jsonfile_path=savepath+'/mask.json'
                with open(jsonfile_path, "w") as f:
                    json.dump(coco_results[iou_type], f)
                # draw_3color_bboxes_on_images(res,savepath,valdata_dir=dataset.root)
            # vis_dataset(res.cocoGt.dataset,savdir='OUTPUT1/GT',TASK_DATASET=dataset.root)
            # vis_dataset(res.cocoDt.dataset, savdir=savepath, TASK_DATASET=dataset.root)
        # logger.info("Evaluating predictions")
    if False:
        mask0=predictions[0].extra_fields['mask']
        import matplotlib.pyplot as plt
        plt.imshow(mask0[0,0,:,:])
        plt.show()
        # for iou_type in iou_types:
        #     with tempfile.NamedTemporaryFile() as f:
        #         file_path = f.name
        #         if output_folder:
        #             file_path = os.path.join(output_folder, iou_type + ".json")
        #         if dataset.coco:
        #             res = evaluate_predictions_on_coco(
        #                 dataset.coco, coco_results[iou_type], file_path, iou_type, MONUGT_being_viewed_as_detection_results
        #             )
        #             # res = evaluate_predictions_on_coco(
        #             #     dataset.coco, coco_results[iou_type], file_path, iou_type
        #             # )
        #             results.update(res)
        #         elif output_folder:
        #             with open(file_path, "w") as f:
        #                 json.dump(coco_results[iou_type], f)
        #
        # logger.info(results)
        # check_expected_results(results, expected_results, expected_results_sigma_tol)
        # if output_folder:
        #     torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results



def sim_sort(predictions,num_of_classes=10):
    rank_preference_value=np.zeros((num_of_classes),dtype=np.int)
    for image_id, prediction in enumerate(predictions):
        if len(prediction) == 0:
            continue
        prediction = prediction.resize((250, 250))
        prediction = prediction.convert("xywh")
        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        box0_record = 0
        repeating_boxes_record = []
        this_box_has_been_repeating=False
        for k, box in enumerate(boxes):
            if box0_record == box[0]:
                if not this_box_has_been_repeating:# it just now starts to repeats,need to create a list to store
                    this_box_repeat_record=[]
                    this_box_has_been_repeating = True
                    this_box_repeat_record.append(k-1)
                    this_box_repeat_record.append(k)
                else: # means this_box_has_been_repeating
                    this_box_repeat_record.append(k)
            elif this_box_has_been_repeating: # repeated ,but now it ends repeating
                this_box_has_been_repeating=False
                box0_record = box[0]
                repeating_boxes_record.append(this_box_repeat_record) # [[5,6],[8,9,10],....]
            else:
                box0_record = box[0]
        if this_box_has_been_repeating :
            repeating_boxes_record.append(this_box_repeat_record)

        for this_box_repeat_record in  repeating_boxes_record: #[8,9,10] in [[5,6],[8,9,10],....]
            scores_of_this_box=[]
            for k in this_box_repeat_record:
                scores_of_this_box.append(scores[k])
            max_score=max(scores_of_this_box)
            max_id=scores_of_this_box.index(max_score)
            rank_preference_value[labels[this_box_repeat_record[max_id]]]+=1
            min_score=min(scores_of_this_box)
            min_id=scores_of_this_box.index(min_score)
            rank_preference_value[labels[this_box_repeat_record[min_id]]] -= 1
    print('rank_preference_value : {}'.format(rank_preference_value[1:]))
    return rank_preference_value
            


def prepare_for_coco_detection(predictions, dataset , USE_SIM_SORT=False,cfg=None):
    # assert isinstance(dataset, COCODataset)
    DO_EVAL=False
    try:
        if 'coco0_grounding_train' in cfg.DATASETS.TRAIN or 'lvis_val' in cfg.DATASETS.TEST or 'odinw_val' in cfg.DATASETS.TEST :
            DO_EVAL=True
    except:
        pass
    if cfg.SOLVER.TEST_WITH_INFERENCE:
        DO_EVAL = True
    if DO_EVAL:
        # assert isinstance(dataset, COCODataset)
        coco_results = []
        total_k=0
        for image_id, prediction in enumerate(predictions):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            # TODO replace with get_img_info?
            image_width = dataset.coco.imgs[original_id]["width"]
            image_height = dataset.coco.imgs[original_id]["height"]
            prediction = prediction.resize((image_width, image_height))
            prediction = prediction.convert("xywh")

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            labels = prediction.get_field("labels").tolist()

            for k, box in enumerate(boxes):
                if labels[k] in dataset.contiguous_category_id_to_json_id:
                    bb = box
                    x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                    coco_results.append(
                        {
                            "image_id": original_id,
                            "category_id": dataset.contiguous_category_id_to_json_id[labels[k]],
                            "bbox": box,
                            "id": total_k,
                            "score": scores[k],
                            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]],
                            "area": bb[2] * bb[3],
                            "iscrowd": 0,
                        })
                    total_k+=1

        return coco_results
    else:
        coco_results = []
        monu_anns_to_record = []
        kcount = 0
        prompt_len = 1
        try:
            prompt_len = dataset.prompt_len
            if prompt_len == 0:  # if --prompt param have no input,it should be an evaluation of a student!!better make sure about this!!!!!!!!!!!!!!!!!!!!
                prompt_len = 1
        except:
            print("if no prompt input,default prompt lenth will be {}!!!!!!!!!!!!!!!!!!!".format(prompt_len))
            pass
        if USE_SIM_SORT:
            sim_sort(predictions)
        for image_id, prediction in enumerate(predictions):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            # TODO replace with get_img_info?
            # image_width = dataset.coco.imgs[original_id]["width"]
            # image_height = dataset.coco.imgs[original_id]["height"]
            prediction = prediction.resize((250, 250))
            prediction = prediction.convert("xywh")

            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()
            labels = prediction.get_field("labels").tolist()
            box0_record = 0
            for k, box in enumerate(boxes):
                if (labels[k] <= prompt_len) and box0_record != box[0]:  #the box0_record will record the last x1 ,used to avoid same boxes
                    coco_results.append(
                        {
                            "image_id": original_id,
                            "category_id": 1,
                            "bbox": box,
                            "area": box[2] * box[3],
                            "id": kcount,
                            "iscrowd": 0,
                            "score": scores[k],
                        })
                    monu_anns_to_record.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": box,
                            "area": box[2] * box[3],
                            "id": kcount,

                            "iscrowd": 0,
                            "score": scores[k],
                        })
                    kcount += 1
                box0_record = box[0]
        import datetime
        now = datetime.datetime.now()
        if 'coco2' in dataset.root:
            jpath='jsonfiles/RAW_GLIP_CONSEP{}.json'.format(now)
        else:
            jpath ='jsonfiles/RAW_GLIP{}.json'.format(now)
        # with open(jpath, "w") as f:
        #     json.dump(monu_anns_to_record, f)
        print('LAST_PREDICT_BBOXS.json successfully created!')
        if cfg.SOLVER.TEST_WITH_INFERENCE:
            THIS_COCO_JSON = dataset.root[:dataset.root.rfind('/')] + '/annotations/instances_val2017.json'
        else:
            THIS_COCO_JSON = 'DATASET/'+cfg.DATASETS.REGISTER.test.ann_file

        f = open(THIS_COCO_JSON, 'r')
        cocogt_dataset = json.load(f)
        cocogt_dataset['annotations']=coco_results
        f.close()
        f = open(jpath, 'w')
        json.dump(cocogt_dataset,f)
        f.close()
        print(
            'first of all, we check AP of glip valset:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        cocoDt = COCO(jpath)
        cocoGt = COCO(THIS_COCO_JSON)
        # print('score seted as 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # for ann in cocoDt.dataset['annotations']:
        #     ann['score'] = 1
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        return jpath


def prepare_for_MONU_GT_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # for image_id, prediction in enumerate(predictions):
    for image_id in range(480):
        original_id = dataset.id_to_img_map[image_id]
        FILENAME_LIST = os.listdir('DATASET/MoNuSACGT/stage1_train')
        FILENAME_LIST.sort()
        THIS_FILENAME = FILENAME_LIST[image_id // 16]
        THIS_CROP_NUM = image_id % 16
        THE_X = THIS_CROP_NUM % 4
        THE_Y = THIS_CROP_NUM // 4
        masks_dir = 'DATASET/MoNuSACGT/stage1_train/' + THIS_FILENAME + '/masks/'
        crop_size = 250
        for instance_mask_file in os.listdir(masks_dir):
            instance_im = io.imread(masks_dir + instance_mask_file)
            BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,
                           THE_Y * crop_size:(THE_Y + 1) * crop_size]
            if np.max(BORROW_PLACE) > 0:
                connection_map = measure.label(BORROW_PLACE)
                connection_map_prop = measure.regionprops(connection_map)
                box = np.array(connection_map_prop[0].bbox).tolist()
                box[0] = max(box[0] - 2, 0)
                box[1] = max(box[1] - 2, 0)
                box[2] = min(box[2] + 2, 250)
                box[3] = min(box[3] + 2, 250)
                a, b, w, h = box  # xmin,ymin,xmax,ymax
                w = box[2] - box[0]
                h = box[3] - box[1]

                box[0] = b
                box[1] = a
                box[2] = h
                box[3] = w
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": 1,
                        "bbox": box,
                    })
    return coco_results


def mask_nms(masks, scores, iou_thresh=0.5):
    """
    Vectorized mask NMS
    masks: BoolTensor [N, H, W]
    scores: List[float] or Tensor[N]
    returns: List[int] - indices to keep
    """
    N, _, H, W = masks.shape
    masks = masks.view(N, -1).float()  # [N, H*W]
    scores = torch.tensor(scores)

    # 计算交集和并集
    inter = torch.matmul(masks, masks.T)  # [N, N]
    areas = masks.sum(dim=1, keepdim=True)  # [N, 1]
    union = areas + areas.T - inter
    ious = inter / (union + 1e-6)  # [N, N]

    # 排序
    idxs = scores.argsort(descending=True)
    keep = []

    suppressed = torch.zeros(N, dtype=torch.bool)
    for i in idxs:
        if suppressed[i]:
            continue
        keep.append(i.item())
        # 抑制所有 IoU > 阈值 的预测
        suppressed = suppressed | (ious[i] > iou_thresh)

    return keep


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np
    import torch.nn.functional as F
    kcount = 0
    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks_float = masks.float()
            masks_resized = F.interpolate(masks_float, size=(image_height, image_width), mode='bilinear', align_corners=False)
            masks = masks_resized > 0.5
            # masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            # masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        # NMS
        keep_indices = mask_nms(masks, scores, 0.3)
        masks = [masks[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        areas = [mask_util.area(rle).item() for rle in rles]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        for k, rle in enumerate(rles):
            if areas[k]!=0:
                coco_results.append(
                    
                        {
                            "image_id": original_id,
                            "category_id": mapped_labels[k],
                            "segmentation": rle,
                            "area": areas[k],
                            "score": scores[k],
                            "iscrowd": 0,
                            "id": kcount,
                        }
                        
                    
                )
                kcount+=1
    
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.to_coco_format()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'keypoints': keypoint,
            'score': scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(
        predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        if prediction.has_field("objectness"):
            inds = prediction.get_field("objectness").sort(descending=True)[1]
        else:
            inds = prediction.get_field("scores").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    if len(gt_overlaps) == 0:
        return {
            "ar": torch.zeros(1),
            "recalls": torch.zeros(1),
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }

    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }

def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json
    pred_coco_dt_dataset=coco_gt.dataset
    pred_coco_dt_dataset['annotations']=coco_results
    with open(json_result_file, "w") as f:
        json.dump(pred_coco_dt_dataset, f)
    # with open(json_result_file, "w") as f:
    #     json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_dt=COCO(json_result_file)
    # coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    if iou_type == 'segm':
    
        from pycocotools import mask as mask_utils
        
        # for ann_id, ann in coco_gt.anns.items():
        #     if isinstance(ann['segmentation'], list):  # polygon
        #         rles = mask_utils.frPyObjects(ann['segmentation'], 250,250)
        #         rle = mask_utils.merge(rles)
        #         ann['segmentation'] = rle
        try:
            metrics = compute_metrics2(coco_gt, coco_dt)
            print(f"Dice (Image Level): {metrics['Dice']:.4f}")
            print(f"Hausdorff Distance: {metrics['HD']:.2f} pixels")
            print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
            print(f"Dice (Object Level): {metrics['Diceobj']:.4f}")
            print(f"AJI: {metrics['AJI']:.4f}")

            jpath ='jsonfiles/RAW_GLIP_SEG_aji{}.json'.format(metrics['AJI'])
            with open(jpath, "w") as f:
                json.dump(pred_coco_dt_dataset, f)
            print('LAST_PREDICT_SEG.json successfully created!')
        except:
            print('some image fail to segment~~~~~~~~~~~~~~~~~~~~')

    if iou_type == 'keypoints':
        coco_gt = filter_valid_keypoints(coco_gt, coco_dt)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    print_mean_iou = True
    if print_mean_iou:
        all_ious = []
        for key in coco_eval.ious.keys():
            item = coco_eval.ious[key]
            try:
                thre05_item = item[item > 0.5]
                for an_iou in thre05_item:
                    all_ious.append(an_iou)
            except:
                pass
        all_ious = np.array(all_ious)
        print('mean iou of TP(threshold 0.5) = {}'.format(np.mean(all_ious)))
    coco_eval.summarize()
    if iou_type == 'bbox':
        summarize_per_category(coco_eval, json_result_file.replace('.json', '.csv'))
    return coco_eval

# def evaluate_predictions_on_coco(
#         coco_gt, coco_results, json_result_file, iou_type="bbox", realgt=None
# ):
#     import json
#
#     with open(json_result_file, "w") as f:
#         json.dump(coco_results, f)
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval
#
#     coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
#     with open(json_result_file, "w") as f:
#         json.dump(realgt, f)
#     realgt2 = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
#     # realgt2 = COCO('MONUGT.json') if coco_results else COCO()
#     # coco_dt = coco_gt.loadRes(coco_results)
#     if iou_type == 'keypoints':
#         coco_gt = filter_valid_keypoints(coco_gt, coco_dt)
#     coco_eval = COCOeval(realgt2, coco_dt, iou_type)
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#     if iou_type == 'bbox':
#         summarize_per_category(coco_eval, json_result_file.replace('.json', '.csv'))
#     return coco_eval


def summarize_per_category(coco_eval, csv_output=None):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        result_str = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ], '. \
            format(titleStr, typeStr, iouStr, areaRng, maxDets)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
            # cacluate AP(average precision) for each category
            num_classes = len(p.catIds)
            avg_ap = 0.0
            for i in range(0, num_classes):
                result_str += '{}, '.format(np.mean(s[:, :, i, :]))
                avg_ap += np.mean(s[:, :, i, :])
            result_str += ('{} \n'.format(avg_ap / num_classes))
        return result_str

    id2name = {}
    for _, cat in coco_eval.cocoGt.cats.items():
        id2name[cat['id']] = cat['name']
    title_str = 'metric, '
    for cid in coco_eval.params.catIds:
        title_str += '{}, '.format(id2name[cid])
    title_str += 'avg \n'

    results = [title_str]
    results.append(_summarize())
    results.append(_summarize(iouThr=.5, maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='small', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='medium', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='large', maxDets=coco_eval.params.maxDets[2]))

    with open(csv_output, 'w') as f:
        for result in results:
            f.writelines(result)


def filter_valid_keypoints(coco_gt, coco_dt):
    kps = coco_dt.anns[1]['keypoints']
    for id, ann in coco_gt.anns.items():
        ann['keypoints'][2::3] = [a * b for a, b in zip(ann['keypoints'][2::3], kps[2::3])]
        ann['num_keypoints'] = sum(ann['keypoints'][2::3])
    return coco_gt


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
