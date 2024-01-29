from __future__ import annotations

import cv2
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    get_track_id,
    vis_pose_result,
)

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30


def show_image(image, bboxes=None):
    if bboxes is not None:
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


class DetectionModel(object):
    def __init__(self, device):
        
        # ViTPose
        pose_model_cfg = osp.join(VIT_DIR, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'vitpose-h-multi-coco.pth')
        self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # YOLO
        bbox_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x.pt')
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        assert scale > 0, f"scale: {scale}"
        return np.array([[cx, cy, scale]])

    def pose_estimation(self, img, bboxes):
        # keypoints detection
        pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results=bboxes,
                format='xyxy',
                return_heatmap=False,
                outputs=None)

        # manual person idendification
        assert len(pose_results) == len(bboxes), f"should predict one pose per bbox. but len(pose_results): {len(pose_results)}, len(bboxes): {len(bboxes)}"
        for pose_result,bbox in zip(pose_results, bboxes):
            xyxy = pose_result['bbox']
            x0,y0,x1,y1 = xyxy
            assert x0 < x1 and y0 < y1, f"xyxy: {xyxy}"
            pose_bbox = self.xyxy_to_cxcys(xyxy)
            bb_bbox = self.xyxy_to_cxcys(np.array(bbox['bbox']))
            assert np.isclose(bb_bbox, pose_bbox).all(), f"bb_bbox: {bb_bbox}, pose_bbox: {pose_bbox}"

            _id = bbox['id']
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bb_bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'][None])

        self.frame_id += 1

    def track(self, img, fps, length, bboxes=None):
        
        # bbox detection
        if bboxes is None:
            bboxes = self.bbox_model.predict(
                img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False)[0].boxes.xyxy.detach().cpu().numpy()

        # show_image(img, bboxes)
        bboxes = [{'bbox': bbox} for bbox in bboxes]

        # keypoints detection
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None)
        
        # person identification
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        
        for pose_result in pose_results:
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'][None])

        self.frame_id += 1
        self.pose_results_last = pose_results

    def process(self, fps):
        for key in ['id', 'frame_id']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        for key in ['keypoints', 'bbox']:
            self.tracking_results[key] = np.concatenate(self.tracking_results[key], axis=0)
            
        output = defaultdict(dict)
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            output[_id]['features'] = []
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output