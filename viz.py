import os
import json
import yaml
import sys
import time
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


from jrdb_split import TRAIN, TEST


# COCO 2d / 3d
COCO_CONNECTIVITY = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]
H36M_CONNECTIVITY = [ (0, 1), (1, 2), (2, 3), (0, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (10, 13), (13, 14), (14, 15), (10, 16), (16, 17), (17, 18), ]


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image



def plot_pose_3d(predicted, gt):
    """
    Given a single 3D predicted pose over time and gt pose, plots them
    both in 3D. Modify this as needed to be more instructive.

    predicted is a np.ndarray
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if predicted is not None:
        draw_pose_3d(predicted, ax, color='b')
    if gt is not None:
        draw_pose_3d(gt, ax, color='r')

    stuff = np.concatenate([predicted, gt], axis=2)
    ax_set_up(ax, stuff)

    fig_image = fig_to_array(fig)
    plt.close('all')

    return fig_image


def draw_pose_3d_single_frame(pose, ax, color, text=False, azim=0, elev=45, connectivities=None):
    """
    Draw a single pose for a given frame index.
    pose: (num_peds, num_joints, 3)
    """
    connectivity = connectivities if connectivities is not None else COCO_CONNECTIVITY
    ax.view_init(azim=azim, elev=elev)
    ax_set_up(ax, pose.reshape(-1, pose.shape[-1]))
    for ped_i in range(pose.shape[0]):  # for each ped
        vals = pose[ped_i]
        for j1, j2 in connectivity:
            x = np.array([vals[j1, 0], vals[j2, 0]])
            y = np.array([vals[j1, 1], vals[j2, 1]])
            z = np.array([vals[j1, 2], vals[j2, 2]])
            ax.plot(x, y, z, lw=2, color=color)

        # label joints with index of joint
        if text:
            for i in range(vals.shape[0]):
                ax.text(vals[i, 0], vals[i, 1], vals[i, 2], str(i))


def ax_set_up(ax, stuff):
    dim_min_x,dim_min_y,dim_min_z = stuff.min(axis=0)[0]
    dim_max_x,dim_max_y,dim_max_z = stuff.max(axis=0)[0]
    ax.set_xlim(dim_min_x, dim_max_x)
    ax.set_ylim(dim_min_y, dim_max_y)
    ax.set_zlim(dim_min_z-0.05, dim_max_z+0.05)


def show_image(image, bboxes=None, poses_gt=None, poses_est=None, connectivities=None):
    """
    bboxes: only the gt
    poses_gt: only the gt
    poses_est: only the estimated poses
    """
    if connectivities is None:
        connectivities = COCO_CONNECTIVITY
    if bboxes is not None:
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    if poses_gt is not None:
        draw_pose_2d_single_frame(poses_gt, image, color=(0, 0, 255), connectivities=connectivities)
    if poses_est is not None:
        draw_pose_2d_single_frame(poses_est, image, color=(255, 0, 0), connectivities=connectivities)
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
