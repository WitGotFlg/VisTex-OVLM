#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["vis"]

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
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 4)

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
def vis_dataset(dataset,savdir='TOSHOW'):#input json_format dataset
    import skimage.io as io
    for image_id in range(480):
        img = cv2.imread('datasets/COCO/val2017/%012d.jpg' % (image_id))
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
