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


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


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


def modify_7dof_pose(pos_ori):
    # Position transformation: swap Y and Z, negate new Y
    position, orientation = pos_ori[:3], pos_ori[3:]
    x, y, z = position
    new_position = np.array([x, -z, y])

    # Orientation transformation
    # Rotate 90 degrees around X (to swap Y and Z)
    rotate_x = Quaternion(axis=[1, 0, 0], angle=np.pi / 2)
    # Rotate 180 degrees around new Z (to negate new Y)
    rotate_z = Quaternion(axis=[0, 0, 1], angle=np.pi)
    new_orientation = rotate_z * rotate_x * orientation

    return (*new_position, *new_orientation)


def rotate_trajectory(trajectory):
    """
    Rotate a trajectory so that the average direction of the first two segments points in the positive x direction.

    :param trajectory: List of (x, y) tuples representing the trajectory points.
    :return: Rotated list of (x, y) tuples.
    """

    def calculate_angle(v1, v2):
        """
        Calculate the angle between two vectors v1 and v2.
        """
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(dot_product)
        return angle

    # Calculate the average direction of the first two segments
    segment_1 = np.array(trajectory[1]) - np.array(trajectory[0])
    segment_2 = np.array(trajectory[2]) - np.array(trajectory[1])
    average_direction = (segment_1 + segment_2) / 2

    # Calculate the angle to rotate
    angle_to_rotate = calculate_angle(average_direction, np.array([1, 0]))

    # Determine the direction of rotation (clockwise or counter-clockwise)
    if np.cross(segment_1, segment_2) > 0:
        angle_to_rotate = -angle_to_rotate

    # Create a rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_to_rotate), -np.sin(angle_to_rotate)],
        [np.sin(angle_to_rotate), np.cos(angle_to_rotate)]
    ])

    # Apply rotation to all points in the trajectory
    rotated_trajectory = [np.dot(rotation_matrix, np.array(point)).tolist() for point in trajectory]

    return rotated_trajectory



def main(scene, args):
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
    egomotion = []
    for cam_num in [0, 2, 4, 6, 8]:
        if use_droid:
            try:
                egomotion.append(np.load(f"../DROID-SLAM/reconstructions/{scene}_image{cam_num}/traj_est.npy"))  # (num_frames, 7)
            except:
                print(f"no DROID-SLAM results for {scene} image{cam_num}")
                continue
        else:
            try:
                egomotion.append(joblib.load(f"../viz/wham-demo/{scene}_image_{cam_num}/slam_results.pth"))
            except:
                print(f"no dvpo results for {scene} image{cam_num}")
                continue

    egomotions_all = np.stack(egomotion, axis=0)
    egomotion = egomotions_all.mean(0)

    additional_yaw = np.arange(5) * np.pi / 5
    assert np.all(additional_yaw < 2 * np.pi), f"additional_yaw should be < 2 pi but is: {additional_yaw / np.pi} pi"
    rotate_egomotion = np.array([[[np.cos(yaw), np.sin(yaw), 0],
                                 [-np.sin(yaw), np.cos(yaw), 0],
                                 [0,0,1]] for yaw in additional_yaw])
    print(f"rotate_egomotion: {rotate_egomotion.shape}")
    scale = 7.5

    ego_positions_all = egomotions_all[:,:,[2,0,1]] * scale
    print(f"ego_positions_all: {ego_positions_all.shape}")
    ego_positions_all[:,:, 1] = -ego_positions_all[:,:, 1]  # negate prev-x-now-y
    ego_positions_all[:,:, 2] = -ego_positions_all[:,:, 2]  # negate prev-y-now-z

    fig = plt.figure()
    # axes = fig.add_subplot(221, projection='3d')
    axes = [fig.add_subplot(221, projection='3d'), fig.add_subplot(222, projection='3d'),
            fig.add_subplot(223, projection='3d'), fig.add_subplot(224, projection='3d')]

    # four different views
    colors = ['r', 'g', 'b', 'c', 'm']
    for ax_i, (ax, (elev, azim)) in enumerate(zip(axes, zip([0, 15, 45, 70], [0, 40, 70, 90]))):
        for ego_pos_i, ego_position in enumerate(ego_positions_all):
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax_set_up(ax, stuff=ego_positions_all.reshape(-1, 3))
            ax.plot(*ego_position.T, color=colors[ego_pos_i], label=f"{ego_pos_i}")
        if ax_i == 2:
            ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.8))

    ego_traj_save_dir = f'../viz/jrdb_egomotion_{droid_or_dvpo}'
    if not os.path.exists(ego_traj_save_dir):
        os.makedirs(ego_traj_save_dir)
    num_cams_available = ego_positions_all.shape[0]
    fig.savefig(f'{ego_traj_save_dir}/{scene}_3d_{num_cams_available}_cams.png')
    plt.close(fig)

    print("done saving plot of egomotions from all cameras")
    return

    # (num_frames, 7)
    assert len(image_list) == len(egomotion), f"len(image_list): {len(image_list)}, len(egomotion): {len(egomotion)}"

    scale = 7.5  # figure out a better way to calculate this

    # swap y and z, then swap x and y, negate new y. we are going from:
    ''' rotate the axes from DROID-SLAM coords to BEV coords
       after:                            before:
                   up z                       z front                          
                      ^   x front            /                                  
                      |  /                  /                          
                      | /                  0 ------> x right                          
       left y <------ 0                    |                          
                                           |      
                                           v      
                                      down y      
    what I am doing right now:
    z becomes x, y becomes z, x becomes -y
    '''
    ego_positions = egomotion[:,[2,0,1]] * scale
    ego_positions[:, 1] = -ego_positions[:, 1]  # negate prev-x-now-y
    ego_positions[:, 2] = -ego_positions[:, 2]  # negate prev-y-now-z
    ego_rotations = egomotion[:, 3:]

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
    ego_positions = ego_positions[:,:2]
    # confirm same length as existing trajectories
    assert len(ego_positions) == len(df['frame'].unique()), \
        f"len(positions): {len(ego_positions)}, len(df['frame'].unique()): {len(df['frame'].unique())}"

    def rotation_matrix(orientation, additional_yaw=0):
        # (using camera coords, the pitch is the rotation along the x-z plane, around the y-axis
        yaw = Quaternion(*orientation).yaw_pitch_roll[1]
        return np.array([[np.cos(yaw+additional_yaw), np.sin(yaw+additional_yaw)],
                         [-np.sin(yaw+additional_yaw), np.cos(yaw+additional_yaw)]])  # negate, the y-axis points downward

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
        for _, row in trajectories.iterrows():
            ax.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
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

        # save fig
        # if t == 0:
        #     plt.savefig(f'{ego_traj_save_dir}/{scene}_temp_2d.png')
        plt.tight_layout()
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
    argparser.add_argument('--use_droid', action='store_true')
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
        scene = 'bytes-cafe-2019-02-07_0'#'tressider-2019-04-26_3'
        main(scene, args)
