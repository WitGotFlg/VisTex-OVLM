#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
__all__ = ["vis"]
def draw_3color_bboxes_on_images(cocoEval, savedir,valdata_dir='DATASETS/COCO/val2017',THRE=0.5):
    # 检查保存结果的文件夹是否存在，如果不存在则创建
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 获取所有图片的ID
    image_ids = cocoEval.params.imgIds
    filename_key = 'file_name'
    for iter,image_id in enumerate(image_ids):
        # 获取当前图片的信息
        image_id=int(image_id)
        image_info = cocoEval.cocoGt.loadImgs(image_id)[0]
        try:
            image_path = os.path.join(image_info[filename_key])
        except:
            filename_key = 'filename'
            image_path = os.path.join(image_info[filename_key])

        image = cv2.imread(valdata_dir+'/'+image_path)
        if image is None:
            image = cv2.imread(valdata_dir + '/%012d.jpg'%image_id)
        gt_results = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=image_id))
        # 获取当前图片的预测结果
        pred_results = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=image_id))
        # pred_bboxes = [result for result in pred_results if result['image_id'] == image_id]

        # 绘制预测结果的bbox
        this_image_pred_ious=cocoEval.ious[(image_id,1)]
        DETECTED_GTBOX_ID=[]
        for box_id in range(min(100,this_image_pred_ious.shape[0])):
            USE_THIS_BOX=True
            box_ious=this_image_pred_ious[box_id,:]
            possible_gtbox_id=np.argmax(box_ious)
            try:
                this_box=cocoEval.cocoDt.anns[cocoEval.evalImgs[iter]['dtIds'][box_id]]
            except:
                print(box_id)
            if np.max(box_ious)>=0.5 :
                if possible_gtbox_id not in DETECTED_GTBOX_ID:
                    color = (51, 204, 51)  # 绿色
                    DETECTED_GTBOX_ID.append(possible_gtbox_id)
                else:
                    USE_THIS_BOX = False
            elif this_box['score']>THRE:
                color = (0, 0, 255)  # 红色
            else:
                USE_THIS_BOX=False
            if this_box['area']>60000:
                USE_THIS_BOX = False
            # 绘制bbox
            if USE_THIS_BOX:
                x, y, w, h = this_box['bbox']
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        for gtbox_id,gt_result in enumerate(gt_results):
            if gtbox_id not in DETECTED_GTBOX_ID:
                bbox = gt_result['bbox']
                color = (0, 255, 255)  # 黄色
                # 绘制bbox
                x, y, w, h = bbox
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        # 绘制未被预测到的GT实例的bbox
        # for gtbox_id,gt_result in enumerate(gt_results):
        #     gtbox_ious=this_image_pred_ious[:,gtbox_id]
        #     if np.max(gtbox_ious)<0.5:
        #         bbox = gt_result['bbox']
        #         color = (0, 255, 255)  # 黄色
        #         # 绘制bbox
        #         x, y, w, h = bbox
        #         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)

        # 保存绘制结果的图片
        save_path = os.path.join(savedir, '%012d.jpg'%image_id)
        cv2.imwrite(save_path, image)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,col=[255,255,255]):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2]+box[0])
        y1 = int(box[3]+box[1])

        # color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def vis_old(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2]+box[0])
        y1 = int(box[3]+box[1])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def vis_dataset(dataset,savdir='TOSHOW',TASK_DATASET='datasets/COCO/val2017'):#input json_format dataset
    import skimage.io as io
    for image_id in range(480):
        img = cv2.imread(TASK_DATASET+'/%012d.jpg' % (image_id))
        if img is None:
            continue
        # img = cv2.cvtColor(imori, cv2.COLOR_BGR2RGB)
        boxes=[ann['bbox'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        try:
            scores=[ann['score'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        except:
            scores=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        cls_ids=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        this_image_vis=vis(img, boxes, scores, cls_ids, conf=0.3, class_names=None)
        # plt.imshow(this_image_vis)
        # plt.show()
        cv2.imwrite('%s/%012d.jpg' % (savdir,image_id),this_image_vis)
def vis_multi_dataset(dataset,dataset2,savdir='TOSHOW'):#input json_format dataset
    import skimage.io as io
    for image_id in range(480):
        imori = cv2.imread('datasets/COCO/val2017/%012d.jpg' % (image_id))
        if imori is None:
            continue
        img = cv2.cvtColor(imori, cv2.COLOR_BGR2RGB)
        boxes=[ann['bbox'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        try:
            scores=[ann['score'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        except:
            scores=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        cls_ids=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        this_image_vis=vis(img, boxes, scores, cls_ids, conf=0.2, class_names=None)
        boxes = [ann['bbox'] for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        try:
            scores = [ann['score'] for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        except:
            scores = [1 for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        cls_ids = [1 for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        this_image_vis = vis(img, boxes, scores, cls_ids, conf=0.1, class_names=None,col=[0,255,0])
        # plt.imshow(this_image_vis)
        # plt.show()
        cv2.imwrite('%s/%012d.jpg' % (savdir,image_id),this_image_vis)

_COLORS = np.array(
    [
        0.000, 0, 1,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
