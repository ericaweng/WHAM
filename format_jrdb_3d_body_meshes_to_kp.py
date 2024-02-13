"""
1. get the 3d keypoints from WHAM 3d body meshes using SMPL for JRDB
2. visualize them
2. remove noisy detections using heuristics (body orientation, body verticality, bounding box visibility)
3. format them for use in AgentFormer
"""

import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import yaml
import trimesh
from tqdm import tqdm
import joblib
import cv2
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from jrdb_split import TRAIN


def main(scene, args):
    # for scene in TRAIN:
    # for cam_num in [0, 2, 4, 6, 8]:
    scene = 'gates-ai-lab-2019-02-08_0'
    cam_num = 0
    path = f'../viz/wham-demo/{scene}_image{cam_num}/results.pth'
    results = joblib.load(path)

    print(f"body_results: {body_results.keys()}")

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format JRDB 3D body meshes into AgentFormer-consumable format.')
    parser.add_argument('--dataroot_poses', type=str, default='/home/eweng/wham/datasets/wham/annotations')
    parser.add_argument('--dataroot_3d_body_meshes', type=str, default='/home/eweng/wham/datasets/wham/3d_body_meshes')
    parser.add_argument('--output_dir', type=str, default='/home/eweng/wham/datasets/wham/3d_body_meshes_agentformer')
    parser.add_argument('--input_traj_dir', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw')
    parser.add_argument('--output_traj_dir', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb_adjusted')
    parser.add_argument('--scene', type=str, default='gates-ai-lab-2019-02-08_0')
    parser.add_argument('--cam_num', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args.scene, args)
