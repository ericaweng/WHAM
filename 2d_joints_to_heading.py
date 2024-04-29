""" get 2d joint motion embeddings  """

import os
import json
import yaml
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.models.layers import MotionEncoder
from lib.models import build_motion_encoder, build_body_model
from custom2 import CustomDataset

from jrdb_split import TRAIN, TEST


# COCO 2d / 3d
COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]

def show_image(image, bboxes=None, poses_gt=None, poses_est=None):
    """
    bboxes: only the gt
    poses_gt: only the gt
    poses_est: only the estimated poses
    """
    if bboxes is not None:
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    if poses_gt is not None:
        draw_pose_2d_single_frame(poses_gt, image, color=(0, 0, 255), connectivities=COCO_CONNECTIVITIES_LIST)
    if poses_est is not None:
        draw_pose_2d_single_frame(poses_est, image, color=(255, 0, 0), connectivities=COCO_CONNECTIVITIES_LIST)
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def draw_pose_2d_single_frame(pose, image, color, text=False, connectivities=None):
    """
    Draw a single pose for a given frame index.
    pose: (num_peds, num_joints, 3)
    """
    connectivity = connectivities if connectivities is not None else CONNECTIVITY_DICT
    for ped_i in range(pose.shape[0]):  # for each ped
        vals = pose[ped_i]
        for j1, j2 in connectivity:
            image = cv2.line(image, vals[j1].round().astype(np.int32), vals[j2].round().astype(np.int32), color, 2)

        # label joints with index of joint
        if text:
            for i in range(vals.shape[0]):
                cv2.putText(image, str(i), vals[i])


def run(cfg, output_pth, network, data, smpl):

    # Build dataset

    tracking_results = {}
    for cam_id in [0,2,4,6,8]:
        tracking_results[cam_id] = joblib.load(osp.join(f'{output_pth}_image_{cam_id}', 'tracking_results.pth'))

    dataset = CustomDataset(cfg, data, tracking_results, smpl)

    # run WHAM
    for batch in dataset:
        if batch is None: break

        # data
        _id, x, init_kp = batch

        # inference
        pred_kp3d, motion_context = network(x, init_kp)
        print(f"motion_context: {motion_context.shape}")
        print(f"pred_kp3d: {pred_kp3d.shape}")
        old_motion_context = motion_context.detach().clone()
        print(f"old_motion_context: {old_motion_context.shape}")
        import ipdb; ipdb.set_trace()

        # Store results
        results[_id]['embedding'] = pred['poses_body'].detach().cpu().squeeze(0).numpy()

    # save results
    joblib.dump(results, osp.join(output_pth, 'results.pth'))
    logger.info(f'Save results at {output_pth}')


def main(args):

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    poses_2d = np.load('../AgentFormerSDD/datasets/jrdb_adjusted/poses_2d_for_wham_pose_embedding.npz',
                       allow_pickle=True)['data'].item()

    embeddings = {}
    for scene in poses_2d:
        print(f"scene: {scene}")

        logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
        logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

        output_pth = osp.join(args.output_pth, scene)
        # os.makedirs(output_pth, exist_ok=True)

        # ========= Load WHAM ========= #
        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        network = build_motion_encoder(cfg)
        network.eval()

        # check that data frames is good
        # for ped_id in poses_2d[scene]:
        #     assert np.all(np.diff(sorted(poses_2d[scene][ped_id].keys())) == 1)
        # import ipdb; ipdb.set_trace()

        with torch.no_grad():
            embeddings[scene] = run(cfg, output_pth, network, poses_2d[scene], smpl)

        logger.info('Done !')

        # Output folder
        os.makedirs(output_pth, exist_ok=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='examples/demo_video.mp4',
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='../viz/wham-demo',
                        help='output folder to write results')

    parser.add_argument('--calib', type=str, default=None,
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', '-lo', action='store_true',
                        help='Only estimate motion in camera coordinate if True')

    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the output mesh if True')

    parser.add_argument('--dont_use_gt_poses', dest='use_gt_poses', action='store_false')#true')
    parser.add_argument('--use_all_gt_boxes', action='store_true')

    args = parser.parse_args()
    main(args)
