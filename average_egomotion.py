""" show wham slam results on 2d BEV jrdb data """

import sys
import yaml
import imageio
import pandas as pd

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import time
import argparse
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib

from torch.multiprocessing import Process

import torch.nn.functional as F

from jrdb_split import TRAIN, TEST


def ax_set_up(ax, stuff):
    if stuff is not None:
        center = (stuff.min(axis=0) + stuff.max(axis=0)) / 2
        width_xyz = (stuff.max(axis=0) - stuff.min(axis=0))
        width = width_xyz.max()

        dim_min_x = center[0] - width / 2
        dim_max_x = center[0] + width / 2
        dim_min_y = center[1] - width / 2
        dim_max_y = center[1] + width / 2
        dim_min_z = center[2] - width / 2
        dim_max_z = center[2] + width / 2
        ax.set_xlim(dim_min_x, dim_max_x)
        ax.set_ylim(dim_min_y, dim_max_y)
        ax.set_zlim(dim_min_z, dim_max_z)


def main(scene, args):
    scene_to_scale_dvpo = {'clark-center-2019-02-28_0': 5,
                     'clark-center-2019-02-28_1': 1,
                     'clark-center-intersection-2019-02-28_0': 1,
                     'cubberly-auditorium-2019-04-22_0': 1,  # small amount of rotation
                     'forbes-cafe-2019-01-22_0': 1,
                     'gates-159-group-meeting-2019-04-03_0': 1,
                     'gates-to-clark-2019-02-28_1': 1,  # linear movement
                     'memorial-court-2019-03-16_0': 1,
                     'huang-2-2019-01-25_0': 1,
                     'huang-basement-2019-01-25_0': 1,
                     'meyer-green-2019-03-16_0': 1,  # some rotation and movement
                     'nvidia-aud-2019-04-18_0': 1,  # small amount of rotation
                     'packard-poster-session-2019-03-20_0': 1,  # some rotation and movement
                     'tressider-2019-04-26_2': 1,}

    print(f"scene: {scene}")
    split = 'train' if scene in TRAIN else 'test'

    # load in ego-perspective camera rgb images for plotting
    image_dir = f"jrdb/{split}/images/image_0/{scene}"
    image_list = sorted(os.listdir(image_dir))

    # load all images in image_list
    images = [cv2.imread(os.path.join(image_dir, imfile)) for imfile in image_list]

    # load all egomotions from all cameras
    use_droid = args.use_droid
    droid_or_dvpo = "droidslam" if use_droid else "dvpo"
    egomotions_all = []
    for cam_num in [0,2]:#, 2, 4, 6, 8]:
        if use_droid:
            traj_path = f"../DROID-SLAM/reconstructions/{scene}_image{cam_num}/traj_est.npy"
            try:
                egomotions_all.append(np.load(traj_path))  # (num_frames, 7)
            except:
                print(f"no DROID-SLAM results at {traj_path} ({scene} image{cam_num})")
                continue
        else:
            traj_path = f"../viz/wham-demo/{scene}_image_{cam_num}/slam_results.pth"
            try:
                egomotions_all.append(joblib.load(traj_path))
            except:
                print(f"no dvpo results at {traj_path} ({scene} image{cam_num})")
                continue
    egomotions_all = np.stack(egomotions_all, axis=0)

    # rotate axes
    scale = scene_to_scale_dvpo[scene]
    egomotions_all[...,:3] = egomotions_all[:,:,[2,0,1]] * scale

    # swap y and z, then swap x and y, negate new y. we are going from:
    ''' rotate the axes from DROID-SLAM coords to BEV coords
        before:                       after:
                z front                          up z
               /                                 ^   x front
              /                                  |  /
             0 ------> x right                   | /
             |                    left y <------ 0
             |      
             v      
        down y
    what I am doing right now:
    z becomes x, y becomes z, x becomes -y '''

    egomotions_all[..., 1] = -egomotions_all[..., 1]  # negate prev-x-now-y
    egomotions_all[..., 2] = -egomotions_all[..., 2]  # negate prev-y-now-z

    # rotate by the rotation of the camera on the robot (2 * pi / 5 for 5 cams)
    additional_yaw = -np.arange(5) * np.pi / 7
    for ego_pos_i, (egomotion, yaw) in enumerate(zip(egomotions_all, additional_yaw)):
        rotate_egomotion = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                                     [-np.sin(yaw), np.cos(yaw), 0],
                                     [           0,           0, 1]])
        egomotions_all[ego_pos_i,:,:3] = np.matmul(rotate_egomotion, egomotion[:,:3].T).T

    egomotion_mean = egomotions_all.mean(0)

    #### plot all cams in 3d
    fig = plt.figure()
    # axes = fig.add_subplot(221, projection='3d')
    axes = [fig.add_subplot(221, projection='3d'), fig.add_subplot(222, projection='3d'),
            fig.add_subplot(223, projection='3d'), fig.add_subplot(224, projection='3d')]

    # four different views
    colors = ['r', 'g', 'b', 'c', 'm']
    for ax_i, (ax, (elev, azim)) in enumerate(zip(axes, zip([0, 15, 45, 70], [0, 40, 70, 90]))):
        for ego_pos_i, egomotion in enumerate(egomotions_all):
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax_set_up(ax, stuff=egomotions_all[...,:3].reshape(-1, 3))
            ax.plot(*egomotion[...,:3].T, color=colors[ego_pos_i], label=f"{ego_pos_i}")
        ax.plot(*egomotion_mean[...,:3].T, color='k', label=f"avg")
        if ax_i == 2:
            ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.8))

    ego_traj_save_dir = f'../viz/jrdb_egomotion_{droid_or_dvpo}'
    if not os.path.exists(ego_traj_save_dir):
        os.makedirs(ego_traj_save_dir)
    num_cams_available = egomotions_all.shape[0]
    fig.savefig(f'{ego_traj_save_dir}/{scene}_3d_{num_cams_available}_cams.png')
    plt.close(fig)

    print("done saving plot of egomotions from all cameras")

    # (num_frames, 7)
    assert len(image_list) == egomotions_all.shape[1], f"len(image_list): {len(image_list)}, len(egomotion): {egomotions_all.shape[1]}"

    # okay from here it's plotting the ego-motion adjusted 2d BEV trajectories
    # load BEV 2d trajs
    bev_traj_dir = '/home/eweng/code/PoseFormer/try2/CIWT/data'
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None, usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y']
    df_ego = df.copy()
    assert len(df['frame'].unique()) == len(image_list), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(image_list)}"

    # only keep the first BEV coords of the ego positions
    # ego_positions = egomotion_mean[...,:2]
    # ego_rotations = egomotion_mean[..., 3:]
    ego_positions = egomotions_all[0,...,:2]
    ego_rotations = egomotions_all[0,..., 3:]

    # confirm same length as existing trajectories
    assert egomotions_all.shape[1] == len(df['frame'].unique()), \
        f"egomotions_all: {egomotions_all.shape[1]}, len(df['frame'].unique()): {len(df['frame'].unique())}"

    def rotation_matrix(orientation):
        # (using camera coords, the pitch is the rotation along the x-z plane, around the y-axis
        yaw = Quaternion(*orientation).yaw_pitch_roll[1]
        return np.array([[np.cos(yaw), np.sin(yaw)],
                         [-np.sin(yaw), np.cos(yaw)]])  # negate, the y-axis points downward

    # Apply rotation
    frame_to_ego_rot_mat = {frame: rotation_matrix(ego_rotations[i]) for i, frame in enumerate(df['frame'].unique())}

    def apply_rotation(row):
        frame = row['frame']
        rotation_matrix = frame_to_ego_rot_mat[frame]
        pos = np.array([row['x'], row['y']])
        new_pos = rotation_matrix.dot(pos)
        return new_pos

    df[['x', 'y']] = np.array(df.apply(apply_rotation, axis=1).tolist())

    # Apply translation
    x_value_map = pd.Series(ego_positions[:, 0], index=df['frame'].unique())
    y_value_map = pd.Series(ego_positions[:, 1], index=df['frame'].unique())

    df['x'] = df['x'] + df['frame'].map(x_value_map)
    df['y'] = df['y'] + df['frame'].map(y_value_map)

    # Determine the min and max positions for scaling across all frames
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()
    print("min_x: {}, max_x: {}, min_y: {}, max_y: {}".format(min_x, max_x, min_y, max_y))

    min_x_ego, max_x_ego = df_ego['x'].min(), df_ego['x'].max()
    min_y_ego, max_y_ego = df_ego['y'].min(), df_ego['y'].max()

    # Assign unique colors to each pedestrian ID
    unique_ids = df['id'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_ids)))
    color_dict = dict(zip(unique_ids, colors))

    frames = []
    skip = 3
    items = list(zip(images, ego_positions, df.groupby('frame'), df_ego.groupby('frame')))[::skip]
    for (t, (image, ego_pos, (frame, trajectories), (frame_ego, trajectories_ego))) in tqdm(enumerate(items), total=len(images) // skip):
        fig, (ax1, ax, ax_im) = plt.subplots(1, 3, figsize=(20, 10))
        assert frame == frame_ego, f"frame: {frame}, frame_ego: {frame_ego}"

        # plot camera ego image
        ax_im.imshow(image[...,[2,1,0]])

        # global perspective
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_title(f"scene: {scene}, frame: {frame}")
        # other peds
        for _, row in trajectories.iterrows():  # for each ped
            ax.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
            ax.annotate(int(row['id']), (row['x'], row['y']), fontsize=6)
        # ego agent
        ax.add_artist(plt.Circle(ego_pos[:2], radius=0.5, color='red', fill=True))
        # ego path
        ax.plot(ego_positions[:t*skip:skip, 0], ego_positions[:t*skip:skip, 1], color='red')
        # ego rotation
        rot_mat = frame_to_ego_rot_mat[frame]
        theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0]) / np.pi
        theta = str(round(theta, 2)) + " pi"
        ax.add_artist(plt.Arrow(ego_pos[0], ego_pos[1], 10*rot_mat[0,0], 10*-rot_mat[0,1], width=1, color='red'))
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0], f"ego-agent pos: {round(ego_pos[0],1), round(ego_pos[1],1)}, {theta}", fontsize=12, color='black')

        # ego perspective
        ax1.set_xlim(min_x_ego, max_x_ego)
        ax1.set_ylim(min_y_ego, max_y_ego)
        ax1.set_aspect('equal')
        ax.set_title(f"scene: {scene}, frame: {frame}")
        # other peds
        for _, row in trajectories_ego.iterrows():
            ax1.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
        # ego-agent
        ax1.add_artist(plt.Circle((0,0), radius=0.5, color='red', fill=True))
        # ego rotation
        ax1.add_artist(plt.Arrow(0,0, 10, 0, width=1, color='red'))

        plt.tight_layout()
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # save fig
        # if t == 0:
        #     plt.savefig(f'{ego_traj_save_dir}/{scene}_temp_2d.png')
        #     import ipdb; ipdb.set_trace()

        frames.append(plot_image)
        plt.close(fig)

    fps = 2.5
    output_file_3d = f'/home/eweng/code/viz/jrdb_{droid_or_dvpo}'
    if not os.path.exists(output_file_3d):
        os.makedirs(output_file_3d)
    output_file_3d = f'{output_file_3d}/{scene}.mp4'
    with imageio.get_writer(output_file_3d, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"saved to: {output_file_3d}")

    # save new trajectories
    bev_traj_adjusted_path = f'/home/eweng/code/PoseFormer/jrdb_adjusted/{scene}.txt'
    if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
        os.makedirs(os.path.dirname(bev_traj_adjusted_path))
    df.to_csv(bev_traj_adjusted_path, index=False)


if __name__ == '__main__':
    __spec__ = None
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use_droid', '-ud', action='store_true')
    argparser.add_argument('--mp', action='store_true')
    args = argparser.parse_args()

    # TRAIN only
    with_movement = ['clark-center-2019-02-28_0',
                     'clark-center-2019-02-28_1',
                     'clark-center-intersection-2019-02-28_0',
                     'cubberly-auditorium-2019-04-22_0',  # small amount of rotation
                     'forbes-cafe-2019-01-22_0',
                     'gates-159-group-meeting-2019-04-03_0',
                     'gates-to-clark-2019-02-28_1',  # linear movement
                     'memorial-court-2019-03-16_0',
                     'huang-2-2019-01-25_0',
                     'huang-basement-2019-01-25_0',
                     'meyer-green-2019-03-16_0',  # some rotation and movement
                     'nvidia-aud-2019-04-18_0',  # small amount of rotation
                     'packard-poster-session-2019-03-20_0',  # some rotation and movement
                     'tressider-2019-04-26_2', ]

    no_movement = ['bytes-cafe-2019-02-07_0',
                   'gates-ai-lab-2019-02-08_0',
                   'gates-basement-elevators-2019-01-17_1',
                   'hewlett-packard-intersection-2019-01-24_0',
                   'huang-lane-2019-02-12_0',
                   'jordan-hall-2019-04-22_0',
                   'packard-poster-session-2019-03-20_1',
                   'packard-poster-session-2019-03-20_2',
                   'stlc-111-2019-04-19_0',
                   'svl-meeting-gates-2-2019-04-08_0',
                   'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
                   'tressider-2019-03-16_0',
                   'tressider-2019-03-16_1', ]

    print(f"with movement: {len(with_movement)} no movement: {len(no_movement)}")  # 14,13
    list_of_args = []
    if args.mp:
        import multiprocessing as mp
        mp.set_start_method('spawn')
        for scene in with_movement:
            list_of_args.append((scene, args))
        with mp.Pool(60) as p:
            p.starmap(main, list_of_args)
    else:
        scene = 'clark-center-2019-02-28_0'#'bytes-cafe-2019-02-07_0'##'tressider-2019-04-26_2'#
        main(scene, args)
