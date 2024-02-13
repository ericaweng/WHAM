""" Calculate egomotion for jrdb scenes that have static pedestrians with ranges of static motion.
currently doesn't work """

import imageio
import pandas as pd
import glob

from tqdm import tqdm
import numpy as np
import os
import cv2
import argparse
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib

from jrdb_split import TRAIN, TEST
from vis_utils import visualize_BEV_trajs


def smooth_rotations(rotations, sigma=2):
    """
    Smooth a series of rotations in the x-y plane, given in radians, using a Gaussian window average.
    This function takes into account that rotations loop around at 2*pi.

    :param rotations: Array of shape (n,) containing the rotations to be smoothed, where n is the number of points.
    :param sigma: Standard deviation for Gaussian kernel. Higher values result in smoother trajectories.
    :return: Array of shape (n,) containing the smoothed rotations.
    """
    # Convert rotations to complex numbers representing points on the unit circle
    complex_rotations = np.exp(1j * np.array(rotations))

    # Apply Gaussian filter separately to real and imaginary parts
    real_smoothed = gaussian_filter1d(np.real(complex_rotations), sigma=sigma)
    imag_smoothed = gaussian_filter1d(np.imag(complex_rotations), sigma=sigma)

    # Convert smoothed complex numbers back to angles, ensuring the result is within [0, 2*pi)
    smoothed_rotations = np.mod(np.angle(real_smoothed + 1j * imag_smoothed), 2 * np.pi)

    return smoothed_rotations


def matrix_first_value(A):
    """
    Extend the function to work with arrays of more than 2 dimensions, setting all values in each non-zero gap
    along the 0th dimension to the first non-zero value in that gap.

    :param A: A NumPy array with arbitrary values and any number of dimensions.
    :return: A new NumPy array where each gap of non-zero values along the 0th dimension is replaced
             with the first non-zero value from that gap.
    """
    A_copy = A.copy()  # Create a copy of the original array to modify

    # Function to iterate over all indices of a given shape except for the first dimension
    def iterate_indices(shape):
        return np.ndindex(shape[1:])

    for index in iterate_indices(A.shape):
        first_nonzero_found = False
        for i in range(A.shape[0]):  # Iterate over the 0th dimension
            # Construct the full index including the 0th dimension
            full_index = (i, *index)
            if A[full_index] != 0:
                if not first_nonzero_found:
                    first_nonzero_value = A[full_index]  # Store the first non-zero value in the current gap
                    first_nonzero_found = True
                A_copy[full_index] = first_nonzero_value  # Set current value to the first non-zero value in the gap
            else:
                first_nonzero_found = False  # Reset for the next gap

    return A_copy


def get_mask(all_frames, static_ped_ids):
    # Concatenate all dataframes to find unique pedestrian IDs
    unique_ids = all_frames['id'].unique()
    num_frames = len(all_frames['frame'].unique())
    num_pedestrians = len(unique_ids)

    # Initialize the array and mask
    static_pedestrians = np.zeros((num_frames, num_pedestrians, 2))
    presence_mask = np.zeros((num_frames, num_pedestrians))
    static_peds_init_pose = np.zeros((num_frames, num_pedestrians, 2))

    # Fill in the data
    for frame, df in all_frames.groupby('frame'):
        for index, row in df.iterrows():
            ped_index = np.where(unique_ids == row['id'])[0][
                0]  # Find the index of the pedestrian ID in the unique IDs array
            range = static_ped_ids[row['id']]
            is_single_range_and_in_range = isinstance(range, tuple) and range[0] <= frame < range[1]
            is_mult_range_and_in_range = (isinstance(range, list) and all([isinstance(r, tuple) for r in range])
                                          and np.any([r[0] <= frame < r[1] for r in range]))
            if (static_ped_ids[row['id']] is None
                    or is_single_range_and_in_range
                    or is_mult_range_and_in_range
                    or frame < range):
                static_pedestrians[frame, ped_index] = [row['x'], row['y']]
                presence_mask[frame, ped_index] = 1
                static_peds_init_pose[frame, ped_index] = [row['x'], row['y']]

    return static_pedestrians, static_peds_init_pose, presence_mask


def optimize_rotation_and_delta(static_pedestrians, target, moving_peds, moving_peds_prev_transformed, initial_guess=(0, 0, 0)):
    """
    Optimize the rotation angle and delta trajectory to ensure that static pedestrians remain in their
    starting positions after applying the rotation and translation.

    :param static_pedestrians: Array of shape (n_static, 2) containing the initial positions of static pedestrians
    :param initial_guess: Tuple (rotation, delta_x, delta_y) as the initial guess for the optimization
    :return: Optimized rotation angle and delta trajectory (dx, dy)
    """

    def cost_function(params):
        rotation, delta_x, delta_y = params

        # Rotation matrix
        rotation_matrix = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)]
        ])

        # Apply rotation around (0,0) to the static pedestrians
        rotated_positions = static_pedestrians @ rotation_matrix

        # Apply translation to the rotated positions
        final_positions = rotated_positions + np.array([delta_x, delta_y])

        # Calculate the cost as the sum of squared distances from the initial positions
        static_cost = np.sum((final_positions - target) ** 2)

        # Add a cost for the moving pedestrians to stay in the same place
        if len(moving_peds) == 0:
            return static_cost
        moving_cost = 0
        for i in range(len(moving_peds)):
            moving_peds_pos = moving_peds[i]@ rotation_matrix+ np.array([delta_x, delta_y])
            moving_peds_prev_pos = moving_peds_prev_transformed[i]
            moving_cost += np.sum((moving_peds_pos - moving_peds_prev_pos) ** 2)
        moving_cost = moving_cost / len(moving_peds)

        cost = static_cost + moving_cost * 0.5
        return cost

    # Minimize the cost function
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')

    optimized_rotation, optimized_delta_x, optimized_delta_y = result.x
    return optimized_rotation, optimized_delta_x, optimized_delta_y


def smooth_trajectory(xy_pairs, sigma=2):
    """
    Smooth a trajectory of x, y pairs using a Gaussian window average.

    :param xy_pairs: Array of shape (n, 2) containing the trajectory to be smoothed, where n is the number of points.
    :param sigma: Standard deviation for Gaussian kernel. Higher values result in smoother trajectories.
    :return: Array of shape (n, 2) containing the smoothed trajectory.
    """
    # Extract x and y coordinates
    x = xy_pairs[:, 0]
    y = xy_pairs[:, 1]

    # Apply Gaussian filter separately to x and y coordinates
    x_smoothed = gaussian_filter1d(x, sigma=sigma)
    y_smoothed = gaussian_filter1d(y, sigma=sigma)

    # Combine smoothed coordinates back into an array of pairs
    smoothed_pairs = np.stack((x_smoothed, y_smoothed), axis=-1)

    return smoothed_pairs


def main(scene, scene_to_list_of_static_ped_ids, args):
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
    bev_traj_dir = args.input_traj_dir#'/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None)  # , usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    df_ego = df.copy()


    #####################################################
    #####   calculate egomotion for each timestep   #####
    #####################################################

    static_ped_ids_to_ranges = scene_to_list_of_static_ped_ids[scene]
    peds_groupby = df.groupby('frame')
    static_peds_df = df[df['id'].isin(static_ped_ids_to_ranges)]

    assert len(peds_groupby) == len(images_0), \
        f'len(static_peds_groupby): {len(peds_groupby)}, len(image_list): {len(images_0)}'
    assert tuple(peds_groupby.groups.keys()) == tuple(range(len(images_0))), \
        f"set(static_peds_groupby.groups.keys()): {set(peds_groupby.groups.keys())}, set(range(len(images_0))): {set(range(len(images_0)))}"

    ego_rotations = [0]
    ego_positions = [[0,0]]
    for t, peds_this_frame in peds_groupby:
        if t == 0:
            moving_peds_last_frame = peds_this_frame[~peds_this_frame['id'].isin(static_ped_ids_to_ranges)]
            continue
        static_ped_poses = []
        static_ped_tgts = []
        static_peds_this_frame = peds_this_frame[peds_this_frame['id'].isin(static_ped_ids_to_ranges)]
        moving_peds_this_frame = peds_this_frame[~peds_this_frame['id'].isin(static_ped_ids_to_ranges)]
        moving_peds_this_frame_and_last_frame = pd.merge(moving_peds_this_frame, moving_peds_last_frame, on='id', how='inner', suffixes=['','_last'])
        moving_peds_this_frame = moving_peds_this_frame_and_last_frame[['x', 'y']].values
        moving_peds_last_frame = moving_peds_this_frame_and_last_frame[['x_last', 'y_last']].values
        static_ped_ids = static_peds_this_frame['id'].values
        for static_ped_id in static_ped_ids:
            rr = static_ped_ids_to_ranges[static_ped_id]
            static_curr_pos = static_peds_this_frame[static_peds_this_frame['id'] == static_ped_id][['x', 'y']].values
            if rr is None or isinstance(rr, int) and t <= rr:  # static for all frames
                static_ped_poses.append(static_curr_pos)
                min_frame_this_ped = static_peds_df[static_peds_df['id'] == static_ped_id]['frame'].min()
                static_first_pos = static_peds_df[(static_peds_df["frame"] == min_frame_this_ped) & (static_peds_df['id'] == static_ped_id)][['x', 'y']].values
                assert len(static_first_pos) == 1, f"len(static_first_pos): {len(static_first_pos)}"
                static_ped_tgts.append(static_first_pos)
            elif isinstance(rr, tuple) and rr[0] < t <= rr[1]:  # static for a certain range of frames
                static_ped_poses.append(static_curr_pos)
                static_ped_init_pose = static_peds_df[(static_peds_df["frame"] == rr[0]) & (static_peds_df['id'] == static_ped_id)][['x', 'y']].values
                assert len(static_ped_init_pose) == 1, f"len(static_ped_init_pose): {len(static_ped_init_pose)}"
                static_ped_tgts.append(np.array([ego_positions[rr[0]]]) + static_ped_init_pose)
            elif isinstance(rr, list) and all([isinstance(r, tuple) for r in rr]):  # static for multiple ranges of frames
                for r in rr:
                    if r[0] <= t < r[1]:
                        static_ped_poses.append(static_curr_pos)
                        static_ped_init_pose = static_peds_df[
                            (static_peds_df["frame"] == r[0]) & (static_peds_df['id'] == static_ped_id)][
                            ['x', 'y']].values
                        assert len(
                            static_ped_init_pose) == 1, f"len(static_ped_init_pose): {len(static_ped_init_pose)}"
                        static_ped_tgts.append(np.array([ego_positions[r[0]]]) + static_ped_init_pose)
                        break

        assert len(static_ped_poses) == len(static_ped_tgts), f"len(static_ped_poses): {len(static_ped_poses)}, len(static_ped_tgts): {len(static_ped_tgts)}"
        assert len(static_ped_poses) + len(moving_peds_this_frame) > 0, f"can't do optimization if len(static_ped_poses) ({len(static_ped_poses)}) == 0"
        static_ped_poses = np.concatenate(static_ped_poses)
        static_ped_tgts = np.concatenate(static_ped_tgts)

        assert static_ped_poses.shape[1] == 2, f"static_ped_poses.shape[1]: {static_ped_poses.shape[1]}"
        assert len(static_ped_poses.shape) == 2, f"len(static_ped_poses.shape): {len(static_ped_poses.shape)}"

        # set initial guess to 0 for first frame
        initial_guess = np.array([*ego_positions[-1], ego_rotations[-1]])
        moving_peds_prev_transformed = moving_peds_last_frame @ np.array([[np.cos(ego_rotations[-1]), -np.sin(ego_rotations[-1])],
                                                                            [np.sin(ego_rotations[-1]), np.cos(ego_rotations[-1])]]) + np.array([ego_positions[-1]])
        rotation, delta_x, delta_y = optimize_rotation_and_delta(static_ped_poses, static_ped_tgts,
                                                                 [],[],
                                                                 # moving_peds_this_frame , moving_peds_prev_transformed,
                                                                 initial_guess)
        # sanity checking that the first timestep should put the ego at about 0,0
        # if t == 0:
        #     assert np.isclose(rotation, 0), f"rotation: {rotation}"
        #     assert np.isclose(delta_x, 0), f"delta_x: {delta_x}"
        #     assert np.isclose(delta_y, 0), f"delta_y: {delta_y}"
        ego_rotations.append(rotation)
        ego_positions.append([delta_x, delta_y])
        moving_peds_last_frame = peds_this_frame[~peds_this_frame['id'].isin(static_ped_ids_to_ranges)]

    ego_positions = smooth_trajectory(np.array(ego_positions), sigma=5)
    delta_x, delta_y = ego_positions[:, 0], ego_positions[:, 1]
    ego_rotations = smooth_rotations(ego_rotations, sigma=1)

    # Apply rotation to existing trajectory data
    frame_to_ego_rot_mat = {frame: np.array([
            [np.cos(ego_rotations[i]), -np.sin(ego_rotations[i])],
            [np.sin(ego_rotations[i]), np.cos(ego_rotations[i])]
    ]) for i, frame in enumerate(df['frame'].unique())}

    def apply_rotation(row):
        frame = row['frame']
        rotation_matrix = frame_to_ego_rot_mat[frame]
        pos = np.array([row['x'], row['y']])
        new_pos = pos.dot(rotation_matrix)
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

        # bev_traj_adjusted_path = f'/home/eweng/code/PoseFormer/datasets/jackrabbot/jrdb_adjusted/{scene}.txt'
        bev_traj_adjusted_path = f'{args.output_traj_dir}/{scene}.txt'

        if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
            os.makedirs(os.path.dirname(bev_traj_adjusted_path))
        # print(f"df: {df}")
        # df = df[df['frame'] % skip == 0]
        # print(f"df: {df}")
        df.to_csv(bev_traj_adjusted_path, index=False, header=False, sep=' ')

    ####################
    ##### plotting #####
    ####################
    if args.visualize:
        visualize_BEV_trajs(df, df_ego, ego_positions, ego_rotations, scene, args)

if __name__ == "__main__":
    scene_to_list_of_static_ped_ids = {
            'clark-center-2019-02-28_0': {13: 114, 18: 138, 108: None, 110: None, 152: None, 153: None, 155: None,
                                          150: None,
                                          162: None, 149: None, 154: None},
            'nvidia-aud-2019-04-18_0': {20: None, 27: None, 31: None, 37: None},  # small amount of rotation
            'huang-basement-2019-01-25_0': {1: None, 10: None, 11: None, 17: None,  # 18: None,# 29: None, 6: None,
                                            33: None, 34: None, 45: None, 46: None, },
            # 53: None }, 40: None, 42: None,
    }
    scene_to_list_of_static_ped_ids1 = {
            # ranges can make it work out
            'forbes-cafe-2019-01-22_0': {11: None, 122: None, 124: None, 3: (1335, 1434), 48: (1314, np.inf), 58: None,
                                         115: None},
            'memorial-court-2019-03-16_0': {25: None, 52: None, 47: None},
            'huang-2-2019-01-25_0': {9: [(0, 117), (396, 498), (609, np.inf)],
                                     10: [(0, 117), (396, 498), (780, np.inf)], 16: None, 18: None, 21: None, 24: None,
                                     4: None},
            'packard-poster-session-2019-03-20_0': {},  # {31:300, 41:300},  # some rotation and movement
    }

    # not all scenes have at least 1 static ped
    scene_to_list_of_static_ped_ids0 = {
        'clark-center-2019-02-28_1': {204: None, 208: None, 58: (0, 27), 170: None, 150: None},
        'clark-center-intersection-2019-02-28_0': {103: None, 104: None, 105: None, 110: None, 7: 69, 8: 69},
    }
    # 7:117,8:117,},

    # movement restricted to small timeframes
    scene_to_list_of_static_ped_ids2 = {
            'cubberly-auditorium-2019-04-22_0': {0: None, 1: None},  # small amount of rotation
            # movement: (96, 153), (468, 690)
            # 'tressider-2019-04-26_2': {0:[(201, 420), ]}
            # movement: (201, 258), (903, 1032), (1203, end)
    }

    # no static peds, should process by some other method
    '''
    'gates-to-clark-2019-02-28_1': {},  # linear movement  stops moving at frame 651
    'gates-159-group-meeting-2019-04-03_0': {},
    'meyer-green-2019-03-16_0': {},  # some rotation and movement
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', '-mp', action='store_true')
    parser.add_argument('--save_traj', '-st', action='store_true')
    parser.add_argument('--dont_visualize', '-dv', dest='visualize', action='store_false')
    parser.add_argument('--input_traj_dir', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw')
    parser.add_argument('--output_traj_dir', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb_adjusted')
    parser.add_argument('--output_viz_dir', type=str, default=f'../viz/jrdb_egomotion_static_opt2')
    parser.add_argument('--skip','-s', type=int, default=3)
    args = parser.parse_args()
    __spec__ = None

    if args.mp:
        import multiprocessing as mp

        mp.set_start_method('spawn')
        list_of_args = []
        for scene in scene_to_list_of_static_ped_ids:
            if len(scene_to_list_of_static_ped_ids[scene]) > 0:
                list_of_args.append((scene, scene_to_list_of_static_ped_ids, args))
        with mp.Pool(min(len(list_of_args), mp.cpu_count())) as p:
            p.starmap(main, list_of_args)

    else:
        # scene = 'nvidia-aud-2019-04-18_0'
        # scene = 'huang-basement-2019-01-25_0'
        for scene in scene_to_list_of_static_ped_ids:
            main(scene, scene_to_list_of_static_ped_ids, args)
