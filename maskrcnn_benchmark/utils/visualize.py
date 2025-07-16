#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pycocotools import mask as maskUtils
__all__ = ["vis"]
COLOR_MAP=[(0,0,255),(255,0,0),(0,255,0),(255,255,0),(144,233,144),(128,128,0),(255,192,203),(210,105,30),(100,100,100),(11,1,100),(100,200,100),(200,100,100)]
cls='other. inflammatory. healthy epithelial. dysplastic/malignant epithelial. fibroblast. muscle. endothelial'
def draw_2color_bboxes_on_images(cocoEval, savedir,valdata_dir='DATASETS/COCO/val2017',THRE=0.5,dataset=None,NEED_LABELBOX=False,count_green_ap=False,mask_on=False,GT_MASK_ON=True,mask2contour=False,use_class_colormap=True,vis_every_box=True):
    # 检查保存结果的文件夹是否存在，如果不存在则创建
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 获取所有图片的ID
    image_ids = cocoEval.params.imgIds
    filename_key = 'file_name'
    if count_green_ap:
        TOTAL_green_counter = 0
        TOTAL_GT_counter=0
    for iter,image_id in enumerate(image_ids):
        # 获取当前图片的信息
        if count_green_ap:
            this_im_green_counter=0
        image_id=int(image_id)
        FOUND_DETECT_OBJECT=True
        image_info = cocoEval.cocoGt.loadImgs(image_id)[0]
        try:
            image_path = os.path.join(image_info[filename_key])
        except:
            filename_key = 'filename'
            image_path = os.path.join(image_info[filename_key])
        imagegt = cv2.imread(valdata_dir + '/' + image_path)
        image = cv2.imread(valdata_dir+'/'+image_path)
        mask_to_overlay=np.zeros((image.shape[0],image.shape[1])).astype(bool)
        if image is None:
            image = cv2.imread(valdata_dir + '/%012d.jpg'%image_id)
        gt_results = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=image_id))
        gt_catkeys=[]
        for gt_box in gt_results:
            gt_catkeys.append(gt_box['category_id'])
        # 获取当前图片的预测结果
        pred_results = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=image_id))
        # pred_bboxes = [result for result in pred_results if result['image_id'] == image_id]
        cats=cocoEval.cocoDt.cats
        ORI_DATACATS=dataset.coco.cats# a dict
        selected_class_set=[]
        DETECTED_GTBOX_ID = []
        if mask_on:
            for pred_r in pred_results:
                segmentation = pred_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
                mask = maskUtils.decode(segmentation)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                color=COLOR_MAP[pred_r['category_id']]
                cv2.drawContours(image, [cnt], 0, color, 3)
            save_path = os.path.join(savedir, '%012d.jpg' % image_id)
            cv2.imwrite(save_path, image)
            try:
                for gt_r in gt_results:
                    segmentation = gt_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
                    mask = maskUtils.decode(segmentation)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    color = COLOR_MAP[gt_r['category_id']]
                    cv2.drawContours(imagegt, [cnt], 0, color, 3)
                os.makedirs(savedir[:savedir.rfind('/')]+'/GT', exist_ok=True)
                save_path = os.path.join(savedir[:savedir.rfind('/')]+'/GT', '%012d.jpg' % image_id)
                cv2.imwrite(save_path, imagegt)
            except:
                pass
            continue
        for ori_cat_id,ori_cat_key in enumerate(ORI_DATACATS):
            selected_class_set.append(ORI_DATACATS[ori_cat_key]['name'])
        for cat_id,cat_key in enumerate(cats):
            cat=cats[cat_key]
            if cat['name'] in selected_class_set:
                this_image_pred_ious=cocoEval.ious[(image_id,cat_key)]
                if this_image_pred_ious==[]:
                    continue
                if cat_key not in gt_catkeys:
                    continue
                for box_id in range(min(1000,this_image_pred_ious.shape[0])):
                    USE_THIS_BOX=False
                    box_ious=this_image_pred_ious[box_id,:]
                    possible_gtbox_id=np.argmax(box_ious)
                    ious_of_the_possible_gtbox=this_image_pred_ious[:,possible_gtbox_id]
                    prefered_predbbox_of_the_possible_gtbox=np.argmax(ious_of_the_possible_gtbox)
                    try:
                        this_box=cocoEval.cocoDt.anns[cocoEval.evalImgs[cat_id*4*len(image_ids)+iter]['dtIds'][box_id]]
                        # pred_cat_count=-1
                        # for pred_ann in pred_results:
                        #     if pred_ann['category_id']==cat_key:
                        #         pred_cat_count+=1
                        #         if pred_cat_count==box_id:
                        #             this_box=pred_ann
                        #             break
                    except:
                        print(box_id)
                    if mask_on or vis_every_box:
                        USE_THIS_BOX = True
                    if np.max(box_ious)>=THRE :
                        if mask_on:
                            USE_THIS_BOX = True
                        else:
                            if cocoEval.evalImgs[cat_id*4*len(image_ids)+iter]['gtIds'][possible_gtbox_id] not in DETECTED_GTBOX_ID:
                                color = (51, 204, 51)  # 绿色
                                DETECTED_GTBOX_ID.append(cocoEval.evalImgs[cat_id*4*len(image_ids)+iter]['gtIds'][possible_gtbox_id])
                                if count_green_ap:
                                    this_im_green_counter += 1
                                if box_id==prefered_predbbox_of_the_possible_gtbox:
                                    USE_THIS_BOX = True
                            else:
                                USE_THIS_BOX = False
                    elif this_box['score']>THRE:
                        USE_THIS_BOX = True
                        color = (0, 0, 255)  # 红色
                    else:
                        USE_THIS_BOX = False
                    # if float(this_box['area'])>60000:
                    #     USE_THIS_BOX = False
                    # 绘制bbox
                    if use_class_colormap:
                        color=COLOR_MAP[this_box['category_id']]
                    if USE_THIS_BOX:
                        if mask_on:
                            segmentation = this_box['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]

                            mask = maskUtils.decode(segmentation)
                            mask = mask.squeeze() if mask.ndim == 3 else mask
                            if mask2contour:
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                cnt = contours[0]
                                cv2.drawContours(image, [cnt], 0,color, 1)
                            if not GT_MASK_ON:
                                mask_to_overlay = np.logical_or(mask_to_overlay, mask)
                                # cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                                # cv2.fillPoly(image, [points], color=(0, 128, 0, 128))
                            else:
                                TARGETID=cocoEval.evalImgs[cat_id*4*len(image_ids)+iter]['gtIds'][possible_gtbox_id]
                                for gt_result in gt_results:
                                    if int(gt_result['id']) == TARGETID:
                                        DETECTED_GTBOX=gt_result
                                        x, y, w, h = DETECTED_GTBOX['bbox']
                                        x = int(x)
                                        y = int(y)
                                        points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
                                        before_overlay = image.copy()
                                        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                                        cv2.fillPoly(image, [points], color=color)
                                        # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                                        alpha = 0.3  # 透明度权重
                                        cv2.addWeighted(before_overlay, alpha, image, 1 - alpha, 0, image)
                                        break
                        else:
                            x, y, w, h = this_box['bbox']
                            x = int(x)
                            y = int(y)
                            # if image_id in tumors:
                            #     color=[0,0,255]
                            # else:
                            #     color=[0,255,0]
                            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                        FOUND_DETECT_OBJECT=True
                        text = '{}:{:.1f}%'.format(cat['name'], this_box['score'] * 100)
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                        # paint_Y=int(y)-5
                        # if paint_Y<0:
                        #     paint_Y=int(y)
                        # cv2.putText(image, text, (int(x), int(y)-5), font, 0.4, color, thickness=1)
                        txt_color = (0, 0, 0) if np.mean(color) > 0.5 * 255 else (255, 255, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        txt_size = cv2.getTextSize(text, font, 0.4*(image.shape[1]/500), 1)[0]
                        txt_bk_color = (int(color[0]*0.7),int(color[1]*0.7),int(color[2]*0.7))
                        if NEED_LABELBOX:
                            cv2.rectangle(
                                image,
                                (int(x), int(y) + 1),
                                (int(x) + txt_size[0] + 1, int(y) + int(1.5 * txt_size[1])),
                                txt_bk_color,
                                -1
                            )
                            cv2.putText(image, text, (x, y + txt_size[1]), font, 0.4, txt_color, thickness=1)
        if count_green_ap:
            print('image NO.{} :   detected {} ---- GT all {}'.format(image_id,len(pred_results),len(gt_results)))
            TOTAL_green_counter+=len(pred_results)#this_im_green_counter
            TOTAL_GT_counter+=len(gt_results)
        if mask_on and not mask2contour:
            image = overlay_mask(image, mask_to_overlay, color=color, alpha=0.3)
        # else:
        #     for gtbox_id,gt_result in enumerate(gt_results):
        #         if gt_result['id'] not in DETECTED_GTBOX_ID:
        #             bbox = gt_result['bbox']
        #             color = (0, 255, 255)  # 黄色
        #             # 绘制bbox
        #             x, y, w, h = bbox
        #             x = int(x)
        #             y = int(y)
        #             text = '{}'.format(cats[gt_result['category_id']]['name'])
        #             # font = cv2.FONT_HERSHEY_SIMPLEX
        #             # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        #             # cv2.putText(image, text, (int(x), int(y)+5), font, 0.4, color, thickness=1)
        #             cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        #
        #             txt_color = (0, 0, 0) if np.mean(color) > 0.5 * 255 else (255, 255, 255)
        #             font = cv2.FONT_HERSHEY_SIMPLEX
        #             txt_size = cv2.getTextSize(text, font, 0.4*(image.shape[1]/500), 1)[0]
        #             txt_bk_color = (int(color[0]*0.7),int(color[1]*0.7),int(color[2]*0.7))
        #             if NEED_LABELBOX:
        #                 cv2.rectangle(
        #                     image,
        #                     (int(x), int(y) + 1),
        #                     (int(x) + txt_size[0] + 1, int(y) + int(1.5 * txt_size[1])),
        #                     txt_bk_color,
        #                     -1
        #                 )
        #                 cv2.putText(image, text, (x, y + txt_size[1]), font, 0.4, txt_color, thickness=1)
        #             # 绘制未被预测到的GT实例的bbox
        #             # for gtbox_id,gt_result in enumerate(gt_results):
        #             #     gtbox_ious=this_image_pred_ious[:,gtbox_id]
        #             #     if np.max(gtbox_ious)<0.5:
        #             #         bbox = gt_result['bbox']
        #             #         color = (0, 255, 255)  # 黄色
        #             #         # 绘制bbox
        #             #         x, y, w, h = bbox
        #             #         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
        #
        #             # 保存绘制结果的图片
        save_path = os.path.join(savedir, '%012d.jpg'%image_id)
        if FOUND_DETECT_OBJECT:
            cv2.imwrite(save_path, image)
    if count_green_ap:
        print('accuracy: {}, detected {} ---- GT all {}'.format(TOTAL_green_counter/TOTAL_GT_counter,TOTAL_green_counter,TOTAL_GT_counter))
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
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
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
import numpy as np
import cv2
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.optimize import linear_sum_assignment

import numpy as np
from scipy.optimize import linear_sum_assignment

def get_fast_pq_from_masks(true_masks, pred_masks, match_iou=0.5):
    """
    使用匈牙利算法从 binary mask 列表计算 Panoptic Quality（PQ）。
    参数:
        true_masks: List[np.ndarray]，每个是 shape=(H, W) 的 bool mask。
        pred_masks: List[np.ndarray]，每个是 shape=(H, W) 的 bool mask。
        match_iou: float，匹配所需的最小 IoU 阈值（默认 0.5）。
    返回:
        dq: Detection Quality
        sq: Segmentation Quality
        pq: Panoptic Quality
    """
    assert match_iou >= 0.0, "IoU threshold must be non-negative"

    if len(true_masks) == 0 and len(pred_masks) == 0:
        return 1.0, 1.0, 1.0
    if len(true_masks) == 0:
        return 0.0, 0.0, 0.0
    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0

    num_true = len(true_masks)
    num_pred = len(pred_masks)
    iou_matrix = np.zeros((num_true, num_pred), dtype=np.float64)

    # 计算 pairwise IoU
    for t_idx, t_mask in enumerate(true_masks):
        for p_idx, p_mask in enumerate(pred_masks):
            inter = np.logical_and(t_mask, p_mask).sum()
            if inter > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                iou_matrix[t_idx, p_idx] = inter / (union + 1e-6)

    # 一对一匈牙利匹配，目标是最大化 IoU（等价于最小化 -IoU）
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= match_iou:
            matched_pairs.append((r, c))

    if len(matched_pairs) == 0:
        return 0.0, 0.0, 0.0

    paired_true, paired_pred = zip(*matched_pairs)
    paired_true = np.array(paired_true)
    paired_pred = np.array(paired_pred)
    paired_iou = iou_matrix[paired_true, paired_pred]

    # 计算 TP / FP / FN
    tp = len(paired_true)
    fp = num_pred - tp
    fn = num_true - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum() / (tp + 1e-6)
    pq = dq * sq
    

    return paired_iou.sum(),tp, fp, fn


def get_fast_aji_from_masks(true_masks, pred_masks, height, width):
    """
    直接从 binary mask 列表计算 AJI，不依赖 instance map，不受重叠问题影响
    true_masks: List[np.ndarray]，每个元素是 shape=(H, W) 的 bool mask
    pred_masks: List[np.ndarray]，同上
    """
    # fast check
    if len(pred_masks) == 0:
        return 0.0

    # 创建 true_id/pred_id 和 mask 的映射
    true_id_list = list(range(1, len(true_masks) + 1))
    pred_id_list = list(range(1, len(pred_masks) + 1))
    true_masks_map = [None] + true_masks
    pred_masks_map = [None] + pred_masks

    # 初始化
    pairwise_inter = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float64)
    pairwise_union = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float64)

    for t_idx, t_mask in enumerate(true_masks, start=1):
        # 找到 t_mask 中重叠的预测实例
        for p_idx, p_mask in enumerate(pred_masks, start=1):
            inter = np.logical_and(t_mask, p_mask).sum()
            if inter > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                pairwise_inter[t_idx - 1, p_idx - 1] = inter
                pairwise_union[t_idx - 1, p_idx - 1] = union

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    matched_pred = np.argmax(pairwise_iou, axis=1)
    matched_iou = np.max(pairwise_iou, axis=1)

    paired_true = np.nonzero(matched_iou > 0.0)[0]
    paired_pred = matched_pred[paired_true]

    overall_inter = pairwise_inter[paired_true, paired_pred].sum()
    overall_union = pairwise_union[paired_true, paired_pred].sum()

    # 统计未匹配的 GT 和 Pred
    matched_true_ids = set(paired_true.tolist())
    matched_pred_ids = set(paired_pred.tolist())

    for t_idx in range(len(true_masks)):
        if t_idx not in matched_true_ids:
            overall_union += true_masks[t_idx].sum()
    for p_idx in range(len(pred_masks)):
        if p_idx not in matched_pred_ids:
            overall_union += pred_masks[p_idx].sum()

    return overall_inter , overall_union

def compute_metrics(gt_annotation_path, pred_annotation_path):
    # 初始化 COCO 对象
    coco_gt = COCO(gt_annotation_path)
    coco_pred = COCO(pred_annotation_path)
    img_ids = coco_gt.getImgIds()

    # 全局统计量
    sum_iou_total = 0.0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    total_dice = 0.0
    total_diceobj = 0.0
    total_hd = 0.0
    total_aji = 0.0
    count_pairs = 0
    num_images = len(img_ids)

    for img_id in img_ids:
        # 加载图像尺寸信息
        img_info = coco_gt.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']

        # 加载标注数据
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        # 处理 GT 的掩膜
        gt_masks = []

        for ann in gt_anns:
            segm = ann['segmentation']
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, height, width)
                mask = maskUtils.decode(rle)
                mask = mask.any(axis=2) if mask.ndim == 3 else mask
                gt_masks.append(mask.astype(bool))
            else:
                mask = maskUtils.decode(segm)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                gt_masks.append(mask.astype(bool))

        # 处理 Pred 的掩膜
        pred_masks = []
        for ann in pred_anns:
            segm = ann['segmentation']
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, height, width)
                mask = maskUtils.decode(rle)
                mask = mask.any(axis=2) if mask.ndim == 3 else mask
                pred_masks.append(mask.astype(bool))
            else:
                mask = maskUtils.decode(segm)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                pred_masks.append(mask.astype(bool))

        num_gt = len(gt_masks)
        num_pred = len(pred_masks)
        iou_matrix = np.zeros((num_gt, num_pred))

        for i in range(num_gt):
            for j in range(num_pred):
                intersection = np.logical_and(gt_masks[i], pred_masks[j]).sum()
                union = np.logical_or(gt_masks[i], pred_masks[j]).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0

        # 匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= 0.5]

        # 统计匹配情况
        tp = len(matched_pairs)
        fp = num_pred - tp
        fn = num_gt - tp

        sum_iou_total += sum(iou_matrix[r, c] for r, c in matched_pairs)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Diceobj、HD、AJI
        aji_numerator = 0
        matched_pred_indices = []
        for r, c in matched_pairs:
            gt_mask = gt_masks[r]
            pred_mask = pred_masks[c]
            matched_pred_indices.append(c)

            # Diceobj
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = gt_mask.sum() + pred_mask.sum()
            total_diceobj += 2 * intersection / union if union > 0 else 0

            # Hausdorff
            gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            gt_points = np.vstack([cnt[:, 0, :] for cnt in gt_contours]) if gt_contours else np.zeros((0, 2))
            pred_points = np.vstack([cnt[:, 0, :] for cnt in pred_contours]) if pred_contours else np.zeros((0, 2))
            if gt_points.size > 0 and pred_points.size > 0:
                hd = max(directed_hausdorff(gt_points, pred_points)[0],
                         directed_hausdorff(pred_points, gt_points)[0])
            else:
                hd = max(height, width)
            total_hd += hd

            # AJI numerator
            aji_numerator += intersection

        # AJI denominator
        unmatched_pred_indices = [c for c in range(num_pred) if c not in matched_pred_indices]
        aji_denominator = sum(mask.sum() for mask in gt_masks) + \
                          sum(pred_masks[c].sum() for c in unmatched_pred_indices)
        total_aji += aji_numerator / aji_denominator if aji_denominator > 0 else 0

        count_pairs += tp

        # 图像级 Dice
        if num_gt == 0 and num_pred == 0:
            total_dice += 1.0
        else:
            gt_union = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
            pred_union = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width), bool)
            intersection = np.logical_and(gt_union, pred_union).sum()
            union = gt_union.sum() + pred_union.sum()
            total_dice += 2 * intersection / union if union > 0 else 0.0

    avg_dice = total_dice / num_images if num_images > 0 else 0.0
    avg_diceobj = total_diceobj / count_pairs if count_pairs > 0 else 0.0
    avg_hd = total_hd / count_pairs if count_pairs > 0 else 0.0
    avg_aji = total_aji / num_images if num_images > 0 else 0.0

    if tp_total + fp_total + fn_total == 0:
        pq = 1.0
    else:
        if tp_total == 0:
            sq = 0.0
            rq = 0.0
        else:
            sq = sum_iou_total / tp_total
            rq = tp_total / (tp_total + 0.5 * (fp_total + fn_total))
        pq = sq * rq

    metrics = {
        'Dice': avg_dice,
        'Diceobj': avg_diceobj,
        'HD': avg_hd,
        'AJI': avg_aji,
        'PQ': pq,
    }
    return metrics

def compute_metrics2(gt_annotation_path, pred_annotation_path):
    if isinstance(gt_annotation_path, COCO):
        coco_gt = gt_annotation_path
        coco_pred = pred_annotation_path
    else:
        coco_gt = COCO(gt_annotation_path)
        # coco_pred = COCO(pred_annotation_path)
        with open(gt_annotation_path, 'r') as f:
            gt_data = json.load(f)

        # Load your prediction annotations
        with open(pred_annotation_path, 'r') as f:
            pred_anns = json.load(f)

        # Create a new COCO dataset structure using the GT structure but replacing annotations
        pred_data = {
            'info': gt_data.get('info', {}),
            'licenses': gt_data.get('licenses', []),
            'categories': gt_data.get('categories', []),
            'images': gt_data.get('images', []),
            'annotations': pred_anns  # This replaces the ground truth annotations
        }
        # Save to a temporary file (COCO needs a file path to initialize)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(pred_data, f)
            f.flush()
            # Initialize COCO with the prediction data
            coco_pred = COCO(f.name)
    img_ids = coco_gt.getImgIds()
    category_ids = coco_gt.getCatIds()
    TOTAL_cat_wise_anns_count=0
    avg_aji = 0
    avg_diceobj = 0
    avg_dice = 0
    avg_hd = 0
    avg_pq=0
    for cat_id in category_ids:
        # 全局统计量
        sum_iou_total = 0.0
        tp_total = 0
        fp_total = 0
        fn_total = 0
        total_dice = 0.0
        total_diceobj = 0.0
        total_hd = 0.0
        total_aji = 0.0
        total_pq = 0.0
        total_sq = 0.0
        total_dq = 0.0
        count_pairs = 0
        total_paired_iou =0
        TOTAL_cat_wise_anns_count = 0
        total_fp =0
        total_tp = 0
        total_fn = 0
        num_images = len(img_ids)
        total_overall_inter = 0
        total_overall_union = 0
        cat_wise_anns_count=0
        for img_id in img_ids:
            # 加载图像尺寸信息
            img_info = coco_gt.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']

            # 加载标注数据
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

            # 处理 GT 的多边形分割掩膜
            gt_masks = []
            for ann in gt_anns:
                segm = ann['segmentation']
                THIS_category_id=ann['category_id']
                if THIS_category_id==cat_id:
                    cat_wise_anns_count+=1
                    TOTAL_cat_wise_anns_count+=1
                else:
                    continue
                # 处理多边形格式（单实例可能包含多个多边形）
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
            if cat_wise_anns_count>0:
                pred_masks = []
                for ann in pred_anns:
                    segm = ann['segmentation']
                    score = ann['score']
                    THIS_category_id = ann['category_id']
                    if THIS_category_id!=cat_id:
                        continue
                    if score < 0.3:
                        continue
                    if isinstance(segm, list):
                        rle = maskUtils.frPyObjects(segm, height, width)
                        mask = maskUtils.decode(rle)
                        mask = mask.any(axis=2) if mask.ndim == 3 else mask
                        pred_masks.append(mask.astype(bool))
                    else:
                        mask = maskUtils.decode(segm)
                        mask = mask.squeeze() if mask.ndim == 3 else mask
                        pred_masks.append(mask.astype(bool))
                        # raise ValueError("Pred segmentation 必须是多边形列表")
                # 计算 IoU 矩阵（匈牙利算法匹配用）
                num_gt = len(gt_masks)
                num_pred = len(pred_masks)
                iou_matrix = np.zeros((num_gt, num_pred))

                for i in range(num_gt):
                    gt_mask = gt_masks[i]
                    for j in range(num_pred):
                        pred_mask = pred_masks[j]
                        intersection = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        iou_matrix[i, j] = intersection / union if union > 0 else 0.0

                # 匈牙利算法匹配（最大化 IoU 总和）
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_pairs = []
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= 0.5:  # IoU 阈值设为 0.5
                        matched_pairs.append((r, c))

                # 统计 TP/FP/FN
                tp = len(matched_pairs)
                fp = num_pred - tp
                fn = num_gt - tp

                # 更新 PQ 的全局统计量
                sum_iou = sum(iou_matrix[r, c] for r, c in matched_pairs)
                sum_iou_total += sum_iou
                tp_total += tp
                fp_total += fp
                fn_total += fn

                # 计算 Diceobj 和 HD（仅匹配对）
                for r, c in matched_pairs:
                    gt_mask = gt_masks[r]
                    pred_mask = pred_masks[c]

                    # Dice 系数（对象级别）
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum())
                    total_diceobj += dice

                    # Hausdorff 距离（需要轮廓点）
                    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # 提取轮廓点
                    gt_points = np.vstack([cnt[:,0,:] for cnt in gt_contours]) if gt_contours else np.zeros((0,2))
                    pred_points = np.vstack([cnt[:,0,:] for cnt in pred_contours]) if pred_contours else np.zeros((0,2))

                    # 计算双向 Hausdorff 距离
                    if gt_points.size > 0 and pred_points.size > 0:
                        hd1 = directed_hausdorff(gt_points, pred_points)[0]
                        hd2 = directed_hausdorff(pred_points, gt_points)[0]
                        hd = max(hd1, hd2)
                    else:
                        hd = max(height, width)  # 无轮廓时用图像对角线长度作为默认值
                    total_hd += hd
                    count_pairs += 1

                # 计算图像级 Dice（合并所有掩膜）
                if num_gt == 0 and num_pred == 0:
                    dice_image = 1.0  # 双方均为空掩膜
                else:
                    gt_combined = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
                    pred_combined = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width), bool)
                    intersection = np.logical_and(gt_combined, pred_combined).sum()
                    dice_image = 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (gt_combined.sum() + pred_combined.sum()) > 0 else 0.0
                total_dice += dice_image * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) #in case of TOAL=0
                try:
                    overall_inter , overall_union = get_fast_aji_from_masks(gt_masks, pred_masks, height, width)
                except:
                    overall_inter, overall_union =(1,1)
                # paired_iou ,tp, fp, fn  = get_fast_pq_from_masks(gt_masks, pred_masks)
                # total_paired_iou += paired_iou
                # total_tp += tp
                # total_fp += fp
                # total_fn += fn
                total_overall_inter +=overall_inter
                total_overall_union +=overall_union
        # avg_dq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn + 1e-6)
        # avg_sq = total_paired_iou / (total_tp + 1e-6)
        # avg_pq = avg_dq * avg_sq
        avg_aji += (1/len(category_ids))* total_overall_inter/total_overall_union * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6))
        # 计算最终指标
        # avg_dq = total_dq /num_images if num_images > 0 else 0.0
        # avg_sq = total_sq /num_images if num_images > 0 else 0.0
        # avg_pq = total_pq /num_images if num_images > 0 else 0.0
        # avg_aji = total_aji /num_images if num_images > 0 else 0.0
        avg_dice += (1/len(category_ids)) * (total_dice / num_images) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if num_images > 0 else 0.0
        avg_diceobj += (1/len(category_ids)) * (total_diceobj / count_pairs) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0
        avg_hd += (1/len(category_ids)) * (total_hd / count_pairs) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0

        # 计算 PQ（全景质量）
        if tp_total + fp_total + fn_total == 0:
            pq = 1.0  # 所有实例均正确匹配且无 FP/FN
        else:
            if tp_total == 0:
                sq = 0.0
                rq = 0.0
            else:
                sq = sum_iou_total / tp_total  # 平均匹配质量
                rq = tp_total / (tp_total + 0.5 * (fp_total + fn_total))  # 识别质量
            pq = sq * rq
        avg_pq += (1/len(category_ids)) * pq * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0
        # print(f"pq: {avg_pq:.4f}")
        # print(f"AJI: {avg_aji:.4f}")
        # print(f"avg_dice: {avg_dice:.4f}")
        # print(f"avg_diceobj: {avg_diceobj:.4f}")
        # print(f"avg_hd: {avg_hd:.4f}")
    metrics= {
        'Dice': avg_dice,
        'HD': avg_hd,
        'PQ': avg_pq,
        'Diceobj': avg_diceobj,
        'AJI': avg_aji
    }
    # print(f"pq_new: {avg_pq:.4f}")
    # print(f"dq_new: {avg_dq:.4f}")
    # print(f"sq_new: {avg_sq:.4f}")
    
    # print(f"Hausdorff Distance: {metrics['HD']:.2f} pixels")
    # print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
    # print(f"Dice (Object Level): {metrics['Diceobj']:.4f}")
    return metrics


from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import hausdorff_distance as directed_hausdorff
import cv2
import pycocotools.mask as maskUtils


def compute_metrics2new(gt_annotation_path, pred_annotation_path):
    if isinstance(gt_annotation_path, COCO):
        coco_gt = gt_annotation_path
        coco_pred = pred_annotation_path
    else:
        coco_gt = COCO(gt_annotation_path)
        # coco_pred = COCO(pred_annotation_path)
        with open(gt_annotation_path, 'r') as f:
            gt_data = json.load(f)

        # Load your prediction annotations
        with open(pred_annotation_path, 'r') as f:
            pred_anns = json.load(f)

        # Create a new COCO dataset structure using the GT structure but replacing annotations
        pred_data = {
            'info': gt_data.get('info', {}),
            'licenses': gt_data.get('licenses', []),
            'categories': gt_data.get('categories', []),
            'images': gt_data.get('images', []),
            'annotations': pred_anns  # This replaces the ground truth annotations
        }

        # Save to a temporary file (COCO needs a file path to initialize)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(pred_data, f)
            f.flush()
            # Initialize COCO with the prediction data
            coco_pred = COCO(f.name)
    img_ids = coco_gt.getImgIds()
    category_ids = coco_gt.getCatIds()

    # 初始化按类别存储的统计量
    class_metrics = defaultdict(lambda: {
        'sum_iou_total': 0.0,
        'tp_total': 0,
        'fp_total': 0,
        'fn_total': 0,
        'total_dice': 0.0,
        'total_diceobj': 0.0,
        'total_hd': 0.0,
        'total_overall_inter': 0,
        'total_overall_union': 0,
        'count_pairs': 0,
        'num_images': 0
    })

    # 全局统计量（所有类别）
    global_metrics = {
        'sum_iou_total': 0.0,
        'tp_total': 0,
        'fp_total': 0,
        'fn_total': 0,
        'total_dice': 0.0,
        'total_diceobj': 0.0,
        'total_hd': 0.0,
        'total_overall_inter': 0,
        'total_overall_union': 0,
        'count_pairs': 0,
        'num_images': len(img_ids)
    }

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']

        # 按类别组织标注
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        # 按类别分组
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        for ann in gt_anns:
            gt_by_class[ann['category_id']].append(ann)

        for ann in pred_anns:
            pred_by_class[ann['category_id']].append(ann)

        # 处理每个类别的标注
        for cat_id in set(gt_by_class.keys()).union(pred_by_class.keys()):
            class_metrics[cat_id]['num_images'] += 1

            # 获取当前类别的标注
            cat_gt_anns = gt_by_class.get(cat_id, [])
            cat_pred_anns = pred_by_class.get(cat_id, [])

            # 处理GT和预测的掩膜
            gt_masks = []
            for ann in cat_gt_anns:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))

            pred_masks = []
            for ann in cat_pred_anns:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    pred_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    pred_masks.append(mask.astype(bool))

            # 计算IoU矩阵
            num_gt = len(gt_masks)
            num_pred = len(pred_masks)
            iou_matrix = np.zeros((num_gt, num_pred))

            for i in range(num_gt):
                gt_mask = gt_masks[i]
                for j in range(num_pred):
                    pred_mask = pred_masks[j]
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()
                    iou_matrix[i, j] = intersection / union if union > 0 else 0.0

            # 匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_pairs = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= 0.5:  # IoU阈值
                    matched_pairs.append((r, c))

            # 统计TP/FP/FN
            tp = len(matched_pairs)
            fp = num_pred - tp
            fn = num_gt - tp

            # 更新类别统计量
            sum_iou = sum(iou_matrix[r, c] for r, c in matched_pairs)
            class_metrics[cat_id]['sum_iou_total'] += sum_iou
            class_metrics[cat_id]['tp_total'] += tp
            class_metrics[cat_id]['fp_total'] += fp
            class_metrics[cat_id]['fn_total'] += fn

            # 计算匹配对的指标
            for r, c in matched_pairs:
                gt_mask = gt_masks[r]
                pred_mask = pred_masks[c]

                # Dice系数
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum())
                class_metrics[cat_id]['total_diceobj'] += dice

                # Hausdorff距离
                gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)

                gt_points = np.vstack([cnt[:, 0, :] for cnt in gt_contours]) if gt_contours else np.zeros((0, 2))
                pred_points = np.vstack([cnt[:, 0, :] for cnt in pred_contours]) if pred_contours else np.zeros((0, 2))

                if gt_points.size > 0 and pred_points.size > 0:
                    hd1 = directed_hausdorff(gt_points, pred_points)[0]
                    hd2 = directed_hausdorff(pred_points, gt_points)[0]
                    hd = max(hd1, hd2)
                else:
                    hd = max(height, width)
                class_metrics[cat_id]['total_hd'] += hd
                class_metrics[cat_id]['count_pairs'] += 1

            # 计算图像级Dice
            if num_gt == 0 and num_pred == 0:
                dice_image = 1.0
            else:
                gt_combined = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
                pred_combined = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width),
                                                                                                       bool)
                intersection = np.logical_and(gt_combined, pred_combined).sum()
                dice_image = 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (
                                                                                                         gt_combined.sum() + pred_combined.sum()) > 0 else 0.0
            class_metrics[cat_id]['total_dice'] += dice_image

            # 计算AJI相关统计量
            overall_inter, overall_union = get_fast_aji_from_masks(gt_masks, pred_masks, height, width)
            class_metrics[cat_id]['total_overall_inter'] += overall_inter
            class_metrics[cat_id]['total_overall_union'] += overall_union

            # 更新全局统计量
            global_metrics['sum_iou_total'] += sum_iou
            global_metrics['tp_total'] += tp
            global_metrics['fp_total'] += fp
            global_metrics['fn_total'] += fn
            global_metrics['total_diceobj'] += class_metrics[cat_id]['total_diceobj']
            global_metrics['total_hd'] += class_metrics[cat_id]['total_hd']
            global_metrics['count_pairs'] += class_metrics[cat_id]['count_pairs']
            global_metrics['total_overall_inter'] += overall_inter
            global_metrics['total_overall_union'] += overall_union

    # 计算每个类别的指标
    results = {}
    for cat_id in category_ids:
        metrics = class_metrics[cat_id]
        num_images = metrics['num_images']
        count_pairs = metrics['count_pairs']

        # Dice
        avg_dice = metrics['total_dice'] / num_images if num_images > 0 else 0.0

        # HD
        avg_hd = metrics['total_hd'] / count_pairs if count_pairs > 0 else 0.0

        # PQ
        if metrics['tp_total'] + metrics['fp_total'] + metrics['fn_total'] == 0:
            pq = 1.0
        else:
            if metrics['tp_total'] == 0:
                sq = 0.0
                rq = 0.0
            else:
                sq = metrics['sum_iou_total'] / metrics['tp_total']
                rq = metrics['tp_total'] / (metrics['tp_total'] + 0.5 * (metrics['fp_total'] + metrics['fn_total']))
            pq = sq * rq

        # Diceobj
        avg_diceobj = metrics['total_diceobj'] / count_pairs if count_pairs > 0 else 0.0

        # AJI
        avg_aji = metrics['total_overall_inter'] / metrics['total_overall_union'] if metrics[
                                                                                         'total_overall_union'] > 0 else 0.0

        results[f'class_{cat_id}'] = {
            'Dice': avg_dice,
            'HD': avg_hd,
            'PQ': pq,
            'Diceobj': avg_diceobj,
            'AJI': avg_aji,
            'Num_Images': num_images
        }

    # 计算全局指标（所有类别）
    num_images = global_metrics['num_images']
    count_pairs = global_metrics['count_pairs']

    # Dice
    avg_dice = global_metrics['total_dice'] / num_images if num_images > 0 else 0.0

    # HD
    avg_hd = global_metrics['total_hd'] / count_pairs if count_pairs > 0 else 0.0

    # PQ
    if global_metrics['tp_total'] + global_metrics['fp_total'] + global_metrics['fn_total'] == 0:
        pq = 1.0
    else:
        if global_metrics['tp_total'] == 0:
            sq = 0.0
            rq = 0.0
        else:
            sq = global_metrics['sum_iou_total'] / global_metrics['tp_total']
            rq = global_metrics['tp_total'] / (
                        global_metrics['tp_total'] + 0.5 * (global_metrics['fp_total'] + global_metrics['fn_total']))
        pq = sq * rq

    # Diceobj
    avg_diceobj = global_metrics['total_diceobj'] / count_pairs if count_pairs > 0 else 0.0

    # AJI
    avg_aji = global_metrics['total_overall_inter'] / global_metrics['total_overall_union'] if global_metrics[
                                                                                                   'total_overall_union'] > 0 else 0.0

    results['overall'] = {
        'Dice': avg_dice,
        'HD': avg_hd,
        'PQ': pq,
        'Diceobj': avg_diceobj,
        'AJI': avg_aji,
        'Num_Images': num_images
    }

    return results['overall']

# 辅助函数（需要实现）
import json
import copy
import numpy as np
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
import numpy as np
import cv2
import copy
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
def compute_all_metrics(gt_annotation_path, pred_annotation_path):
    # ==================== 基础指标计算 ====================
    # 初始化COCO对象
    coco_gt = COCO(gt_annotation_path)
    coco_pred = COCO(pred_annotation_path)
    img_ids = coco_gt.getImgIds()

    # 初始化统计量
    metrics = {
        # 分割指标
        'Dice': 0.0, 'HD': 0.0, 'AJI': 0.0, 'Diceobj': 0.0,
        # 检测指标
        'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AR': 0.0
    }

    # ==================== 分割指标计算 ====================
    sum_iou_total = 0.0
    total_aji = 0.0
    count_pairs = 0
    num_images = len(img_ids)

    for img_id in img_ids:
        # 加载图像信息
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        # 加载标注并生成掩膜
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        gt_masks = [_poly_to_mask(ann['segmentation'], h, w) for ann in gt_anns]
        pred_masks = [_poly_to_mask(ann['segmentation'], h, w) for ann in pred_anns]

        # 计算IoU矩阵
        iou_matrix = _compute_iou_matrix(gt_masks, pred_masks)

        # 匈牙利匹配
        matched_pairs, tp, fp, fn = _hungarian_matching(iou_matrix)

        # 更新统计量
        sum_iou_total += sum(iou_matrix[r, c] for r, c in matched_pairs)
        count_pairs += len(matched_pairs)

        # 计算实例级指标
        aji_numerator = 0.0
        for r, c in matched_pairs:
            # Diceobj
            intersection = (gt_masks[r] & pred_masks[c]).sum()
            metrics['Diceobj'] += 2 * intersection / (gt_masks[r].sum() + pred_masks[c].sum())

            # HD
            hd = _calculate_hd(gt_masks[r], pred_masks[c], h, w)
            metrics['HD'] += hd

            # AJI分子
            aji_numerator += (gt_masks[r] & pred_masks[c]).sum()

        # AJI分母
        matched_pred_indices = [c for _, c in matched_pairs]
        unmatched_pred_indices = [c for c in range(len(pred_masks)) if c not in matched_pred_indices]

        aji_denominator = (
                sum(gt_mask.sum() for gt_mask in gt_masks) +
                sum(pred_masks[c].sum() for c in unmatched_pred_indices)
        )
        total_aji += aji_numerator / aji_denominator if aji_denominator > 0 else 0

        # 图像级Dice
        gt_combined = np.logical_or.reduce(gt_masks) if gt_masks else np.zeros((h, w), bool)
        pred_combined = np.logical_or.reduce(pred_masks) if pred_masks else np.zeros((h, w), bool)
        intersection = (gt_combined & pred_combined).sum()
        metrics['Dice'] += 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (
                                                                                                       gt_combined.sum() + pred_combined.sum()) > 0 else 0

    # 归一化指标
    metrics['Dice'] /= num_images
    metrics['Diceobj'] /= count_pairs if count_pairs > 0 else 1
    metrics['HD'] /= count_pairs if count_pairs > 0 else 1
    metrics['AJI'] = total_aji / num_images

    # # ==================== 检测指标计算 ====================
    # # 运行COCO官方评估
    # coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    #
    # # 提取指标
    # metrics['mAP'] = coco_eval.stats[0]
    # metrics['AP50'] = coco_eval.stats[1]
    # metrics['AP75'] = coco_eval.stats[2]
    # metrics['AR'] = coco_eval.stats[8]

    return metrics


# ==================== 工具函数 ====================
def _poly_to_mask(segmentation, height, width):
    """将多边形转换为二值掩膜"""
    rle = maskUtils.frPyObjects(segmentation, height, width)
    mask = maskUtils.decode(rle)
    return mask.any(axis=2) if mask.ndim == 3 else mask


def _compute_iou_matrix(gt_masks, pred_masks):
    """计算IoU矩阵"""
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            intersection = (gm & pm).sum()
            union = (gm | pm).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    return iou_matrix


def _hungarian_matching(iou_matrix, iou_threshold=0.5):
    """匈牙利算法匹配"""
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold]
    tp = len(matched)
    fp = iou_matrix.shape[1] - tp
    fn = iou_matrix.shape[0] - tp
    return matched, tp, fp, fn


def _calculate_hd(gt_mask, pred_mask, img_h, img_w):
    """计算Hausdorff距离"""
    # 提取轮廓 (修复参数传递错误)
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # 生成轮廓点集合 (修复列表推导式括号错误)
    gt_points = np.vstack([c[:, 0, :] for c in gt_contours]) if gt_contours else np.zeros((0, 2))
    pred_points = np.vstack([c[:, 0, :] for c in pred_contours]) if pred_contours else np.zeros((0, 2))

    # 处理空掩膜情况
    if gt_points.size == 0 or pred_points.size == 0:
        return np.sqrt(img_h ** 2 + img_w ** 2)  # 返回图像对角线长度

    # 计算双向Hausdorff距离 (修复max函数调用)
    hd_forward = directed_hausdorff(gt_points, pred_points)[0]
    hd_backward = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_forward, hd_backward)

# gt_ann_path = "/home/data/jy/GLIP/DATASET/coco1/annotations/instances_train2017.json"
# pred_ann_path = "/home/data/jy/GLIP/maskrcnn_benchmark/utils/dice07.json"
# # 使用示例
# metrics = compute_all_metrics(
#     gt_annotation_path=gt_ann_path,
#     pred_annotation_path=pred_ann_path
# )
#
# print(f"""
# 分割指标:
# Dice: {metrics['Dice']:.4f}
# Diceobj: {metrics['Diceobj']:.4f}
# AJI: {metrics['AJI']:.4f}
# HD: {metrics['HD']:.2f} px
#
# 检测指标:
# mAP: {metrics['mAP']:.4f}
# AP50: {metrics['AP50']:.4f}
# AP75: {metrics['AP75']:.4f}
# AR: {metrics['AR']:.4f}
# """)

if __name__ == '__main__':
    # # 使用示例
    gt_ann_path = "/home/data/jy/GLIP/DATASET/coco3s/annotations/instances_val2017.json"
    pred_ann_path ="/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.23546/mask.json"# "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22411/mask.json"
    # # perturbed_data = generate_perturbed_gt(
    # #     gt_json_path=gt_ann_path,
    # #     output_json_path='dice07.json',
    # #     target_dice=0.7
    # # )
    #
    metrics = compute_metrics2(gt_ann_path, pred_ann_path)
    print(f"Dice (Image Level): {metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: {metrics['HD']:.2f} pixels")
    print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
    print(f"Dice (Object Level): {metrics['Diceobj']:.4f}")
    print(f"AJI: {metrics['AJI']:.4f}")
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    将掩码以半透明颜色覆盖到原图上，非掩码区域保持原图不变
    Args:
        image: 原图 (H, W, 3), uint8 格式
        mask: 二值掩码 (H, W), bool类型或0/1
        color: 覆盖颜色，默认绿色 (R, G, B)
        alpha: 透明度 (0为完全透明，1为不透明)
    Returns:
        叠加后的图像 (H, W, 3), uint8
    """
    # 确保输入合法性
    assert image.ndim == 3 and image.shape[2] == 3, "原图必须是HWC格式的三通道图像"
    mask = mask.astype(bool)  # 强制转换为布尔掩码
    if mask.shape != image.shape[:2]:
        raise ValueError("掩码与原图的尺寸不匹配")

    # 转换为浮点类型以便计算
    image_float = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    color_layer = image_float.copy()  # 初始化颜色层为原图

    # 只在掩码区域应用颜色覆盖
    color_layer[mask] = np.array(color) / 255.0  # 将颜色归一化到 [0, 1]

    # 混合原图和颜色层（仅在掩码区域混合）
    blended = image_float.copy()
    blended[mask] = (image_float[mask] * (1 - alpha) +
                     color_layer[mask] * alpha)

    # 转换回uint8格式
    blended = (blended * 255).clip(0, 255).astype(np.uint8)
    return blended
tumors=[2,14,15,16,27,28,29,30,40,41,42,43,44,53,54,55,56,57,58,66,67,68,69,70,71,105,106,107,108,109,118,119,120,121,122,131,132,133,134,135,144,145,146,147,157,158,159,160,170,171,
        261,262,263,264,274,275,276,277,287,220,231,232,233,244,245,246,248,259,271,272,283,284,285,299,317,318,329,330,332,342,343,344,345,356,355,357,358,371,362,363,364,375,376,377,388,389,402,
        436,437,438,449,450,451,452,453,454,464,465,466,476,477,478,489,490,491,492,503,504]