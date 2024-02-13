""" view adjusted egomotion trajectories on 2d BEV jrdb data, adding in rosbag egomotion data
or: view plain trajectories from text file"""

import imageio
import pandas as pd
import glob

from tqdm import tqdm
import numpy as np
import os
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib

from jrdb_split import TRAIN, TEST, WITH_MOVEMENT, NO_MOVEMENT, WITH_MOVEMENT_ADJUSTED
from vis_utils import visualize_BEV_trajs


def main(scene, args):
    # Example usage (Note: Replace these with your actual data)
    print(f"scene: {scene}")
    split = 'train' if scene in TRAIN else 'test'

    # load in ego-perspective camera rgb images for plotting
    images_0 = sorted(glob.glob(f"jrdb/{split}/images/image_0/{scene}/*"))
    images_2 = sorted(glob.glob(f"jrdb/{split}/images/image_2/{scene}/*"))
    images_4 = sorted(glob.glob(f"jrdb/{split}/images/image_4/{scene}/*"))
    images_6 = sorted(glob.glob(f"jrdb/{split}/images/image_6/{scene}/*"))
    images_8 = sorted(glob.glob(f"jrdb/{split}/images/image_8/{scene}/*"))
    assert len(images_0) == len(images_2) == len(images_4) == len(images_6) == len(images_8), \
        (f"len(images_0): {len(images_0)}, len(images_2): {len(images_2)}, len(images_4): {len(images_4)}, "
         f"len(images_6): {len(images_6)}, len(images_8): {len(images_8)}")

    # load BEV 2d trajs
    bev_traj_dir = args.input_traj_dir  # '/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None)  # , usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    df_ego = df.copy()

    if args.adjust_w_egomotion:
        egomotion = np.load(f"jrdb/rosbag_egomotion/{scene}.npy")
        ego_positions = egomotion[:, :2]
        # negate y
        ego_positions[:, 1] = ego_positions[:, 1]
        ego_rotations = egomotion[:, 2]
        delta_x, delta_y = egomotion[-1, :2]

        # Apply rotation to existing trajectory data
        frame_to_ego_rot_mat = {frame: np.array([
                [np.cos(ego_rotations[i]), -np.sin(ego_rotations[i])],
                [np.sin(ego_rotations[i]), np.cos(ego_rotations[i])]
        ]) for i, frame in enumerate(df['frame'].unique())}

        def apply_rotation(row):
            frame = row['frame']
            rotation_matrix = frame_to_ego_rot_mat[frame]
            pos = np.array([row['x'], row['y']])
            new_pos = pos.dot(rotation_matrix.T)
            return new_pos

        df[['x', 'y']] = np.array(df.apply(apply_rotation, axis=1).tolist())

        # Apply translation
        x_value_map = pd.Series(delta_x, index=df['frame'].unique())
        y_value_map = pd.Series(delta_y, index=df['frame'].unique())
        df['x'] = df['x'] + df['frame'].map(x_value_map)
        df['y'] = df['y'] + df['frame'].map(y_value_map)

        # apply rotation
        rot_map = pd.Series(ego_rotations, index=df['frame'].unique())
        df['heading'] = df['heading'] + df['frame'].map(rot_map)

    ####################
    ##### plotting #####
    ####################

    if args.length is None:
        args.length = len(images_0)
    if args.visualize:
        visualize_BEV_trajs(df, images_0, images_2, images_4, images_6, images_8,
                            scene, args)

    # save new trajectories
    if args.save_traj:
        # add egomotion in as an additional pedestrian to the df
        ego_ped_id = 1000
        ego_ped_df = pd.DataFrame(
                {'frame': np.arange(len(images_0)), 'id': ego_ped_id, 'x': ego_positions[:, 0],
                 'y': ego_positions[:, 1],
                 'heading': ego_rotations})
        df = pd.concat([df, ego_ped_df], ignore_index=True)
        df = df.sort_values(by=['frame', 'id']).reset_index(drop=True)

        bev_traj_adjusted_path = f'{args.output_traj_dir}/{scene}.txt'

        if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
            os.makedirs(os.path.dirname(bev_traj_adjusted_path))
        # print(f"df: {df}")
        # df = df[df['frame'] % skip == 0]
        # print(f"df: {df}")
        df.to_csv(bev_traj_adjusted_path, index=False, header=False, sep=' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', '-mp', action='store_true')
    parser.add_argument('--save_traj', '-st', action='store_true')
    parser.add_argument('--dont_visualize', '-dv', dest='visualize', action='store_false')
    parser.add_argument('--input_traj_dir', '-it', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw')
    parser.add_argument('--output_traj_dir', '-ot', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb_adjusted')
    parser.add_argument('--output_viz_dir', '-ov', type=str, default=f'../viz/jrdb_egomotion_rosbag')
    parser.add_argument('--skip','-s', type=int, default=6)
    parser.add_argument('--length', '-l', type=int, default=None)
    parser.add_argument('--adjust_w_egomotion', '-a', action='store_true')
    args = parser.parse_args()
    __spec__ = None

    if args.mp:
        import multiprocessing as mp

        mp.set_start_method('spawn')
        list_of_args = []
        for scene in WITH_MOVEMENT_ADJUSTED:#NO_MOVEMENT + WITH_MOVEMENT_ADJUSTED:
            list_of_args.append((scene, args))
        with mp.Pool(min(len(list_of_args), mp.cpu_count())) as p:
            p.starmap(main, list_of_args)

    else:
        for scene in WITH_MOVEMENT:
            main(scene, args)
