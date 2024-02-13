""" Calculate egomotion for jrdb scenes that have static pedestrians throughout the entire scene via linear optimization method. """

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
    smoothed_rotations = np.mod(np.angle(real_smoothed + 1j*imag_smoothed), 2*np.pi)

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


def optimize_rotation_and_delta(static_pedestrians, target, initial_guess=(0, 0, 0)):
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
        cost = np.sum((final_positions - target) ** 2)
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
    images_0 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_0/{scene}/*"))]
    images_2 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_2/{scene}/*"))]
    images_4 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_4/{scene}/*"))]
    images_6 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_6/{scene}/*"))]
    images_8 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_8/{scene}/*"))]
    assert len(images_0) == len(images_2) == len(images_4) == len(images_6) == len(images_8), \
        f"len(images_0): {len(images_0)}, len(images_2): {len(images_2)}, len(images_4): {len(images_4)}, len(images_6): {len(images_6)}, len(images_8): {len(images_8)}"

    # load BEV 2d trajs
    bev_traj_dir = '/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None)#, usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    df_ego = df.copy()

    # calculate egomotion for each timestep
    static_ped_ids = scene_to_list_of_static_ped_ids[scene]

    if len(static_ped_ids) > 0:
        init_static_poses_df = df[df['id'].isin(static_ped_ids)]
        init_static_poses = init_static_poses_df[init_static_poses_df['frame'] == 0]
        static_peds_groupby = init_static_poses_df.groupby('frame')
        # static_pedestrians_arr, static_peds_init_pose, mask = get_mask(init_static_poses_df, static_ped_ids)

        assert len(static_peds_groupby) == len(images_0), \
            f'len(static_peds_groupby): {len(static_peds_groupby)}, len(image_list): {len(images_0)}'

        # for t, (peds_poses, ped_tgts) in enumerate(zip(static_pedestrians_arr, static_peds_init_pose)):

        rotations = []
        ego_positions = []

        # for t, static_peds_this_frame in static_peds_groupby:
        #     static_ped_poses = []
        #     static_ped_tgts = []
        #     ped_ids = static_peds_this_frame['id'].values
        #     ped_id_to_status = {}
        #     for static_ped in ped_ids:
        #         rr = static_ped_ids[static_ped]
        #         if rr is None:
        #             continue
        #         elif isinstance(rr, tuple) and rr[0] <= t < rr[1]:
        #             static_ped_poses.append([row['x'], row['y']])
        #             static_ped_tgts.append([row['x'], row['y']])
        #         elif isinstance(rr, list) and all([isinstance(r, tuple) for r in rr]) and np.any([r[0] <= t < r[1] for r in rr]):
        #             static_ped_poses.append([row['x'], row['y']])
        #             static_ped_tgts.append([row['x'], row['y']])
        #         elif isinstance(rr, int):
        #             static_ped_poses.append([row['x'], row['y']])
        #             static_ped_tgts.append([row['x'], row['y']])
        #     # set initial guess to 0 for first frame
        #     if t == 0:
        #         initial_guess = (0, 0, 0)
        #     else:
        #         initial_guess = (*ego_positions[-1], rotations[-1])
        #     rotation, delta_x, delta_y = optimize_rotation_and_delta(static_ped_poses, static_ped_tgts, initial_guess)
        #     if t == 0:
        #         assert np.isclose(rotation, 0), f"rotation: {rotation}"
        #         assert np.isclose(delta_x, 0), f"delta_x: {delta_x}"
        #         assert np.isclose(delta_y, 0), f"delta_y: {delta_y}"
        #     rotations.append(rotation)
        #     ego_positions.append([delta_x, delta_y])
        #
        # ego_positions = smooth_trajectory(np.array(ego_positions), sigma=5)
        # delta_x, delta_y = ego_positions[:, 0], ego_positions[:, 1]
        # rotations = smooth_rotations(rotations, sigma=1)

        ####

        for t, static_peds_this_frame in static_peds_groupby:
            joined = static_peds_this_frame.merge(init_static_poses, on='id', suffixes=('','_init'), how='inner')
            curr_ts_poses = joined[['x', 'y']].values
            init_poses = joined[['x_init', 'y_init']].values
            if t == 0:
                initial_guess = (0, 0, 0)
            else:
                initial_guess = (*ego_positions[-1], rotations[-1])
            rotation, delta_x, delta_y = optimize_rotation_and_delta(curr_ts_poses, init_poses, initial_guess)
            if t == 0:
                assert np.isclose(rotation, 0, atol=1e-5), f"rotation: {rotation}"
                assert np.isclose(delta_x, 0, atol=1e-5), f"delta_x: {delta_x}"
                assert np.isclose(delta_y, 0, atol=1e-5), f"delta_y: {delta_y}"
            rotations.append(rotation)
            ego_positions.append([delta_x, delta_y])
        ego_positions = smooth_trajectory(np.array(ego_positions), sigma=5)
        delta_x, delta_y = ego_positions[:, 0], ego_positions[:, 1]
        rotations = smooth_rotations(rotations, sigma=0.5)

        # Apply rotation to existing trajectory data
        frame_to_ego_rot_mat = {frame: np.array([
                    [np.cos(rotations[i]), -np.sin(rotations[i])],
                    [np.sin(rotations[i]), np.cos(rotations[i])]
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
        rot_map = pd.Series(rotations, index=df['frame'].unique())
        df['heading'] = df['heading'] + df['frame'].map(rot_map)
    else:
        ego_positions = np.zeros((len(images_0), 2))
        frame_to_ego_rot_mat = [np.array([[1, 0], [0, 1]])] * len(images_0)

    ##### plotting #####
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

    skip = 1
    if args.visualize:
        frames = []
        items = list(enumerate(zip(ego_positions, df.groupby('frame'), df_ego.groupby('frame'))))[::skip]

        ego_traj_save_dir = f'../viz/jrdb_egomotion_static_opt'
        if not os.path.exists(ego_traj_save_dir):
            os.makedirs(ego_traj_save_dir)

        for (t, (ego_pos, (frame, trajectories), (frame_ego, trajectories_ego))) in tqdm(items[::-1], total=len(images_0) // skip):
            fig, ((ax1, ax, ax_im),(ax_im0,ax_im1,ax_im2)) = plt.subplots(2, 3, figsize=(20, 10))
            assert frame == frame_ego, f"frame: {frame}, frame_ego: {frame_ego}"


            # plot camera ego image((image, image2, image4, image6, image8),
            ax_im.imshow(images_0[t][...,[2,1,0]])  # RGB to BGR
            ax_im.set_title(f"cam 0")
            ax_im2.imshow(images_2[t][...,[2,1,0]])  # RGB to BGR
            ax_im2.set_title(f"cam 2")
            ax_im1.imshow(images_4[t][...,[2,1,0]])  # RGB to BGR
            ax_im1.set_title(f"cam 4")
            ax_im0.imshow(images_6[t][...,[2,1,0]])  # RGB to BGR
            ax_im0.set_title(f"cam 6")

            # global perspective
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            ax.set_title(f"scene: {scene}, frame: {frame}")
            # other static_peds_this_frame
            for _, row in trajectories.iterrows():
                ax.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
                ax.annotate(int(row['id']), (row['x'], row['y']), fontsize=6)
                # plot agent rotations
                ax.arrow(row['x'], row['y'], -np.cos(row['heading']), np.sin(row['heading']), head_width=0.1, head_length=0.1, fc='m', ec='m')

            # ego agent
            ax.add_artist(plt.Circle(ego_pos[:2], radius=0.5, color='red', fill=True))
            # ego path
            ax.plot(ego_positions[:t, 0], ego_positions[:t, 1], color='red')
            # ego rotation
            # rot_mat = frame_to_ego_rot_mat[frame]
            # theta = np.arctan2(-rot_mat[1, 0], rot_mat[0, 0]) / np.pi
            # theta = str(round(theta, 2)) + " pi"
            theta = rotations[t]
            theta_str = str(round(theta, 2)) + " pi"
            ax.add_artist(plt.Arrow(ego_pos[0], ego_pos[1], 5*np.cos(theta), 5*np.sin(theta), width=1, color='red'))
            ax.text(ax.get_xlim()[0], ax.get_ylim()[0], f"ego-agent pos: {round(ego_pos[0],1), round(ego_pos[1],1)}, {theta_str}", fontsize=12, color='black')

            # ego perspective
            ax1.imshow(images_8[t][...,[2,1,0]])  # RGB to BGR
            ax1.set_title(f"cam 8")
            # ax1.set_xlim(min_x_ego, max_x_ego)
            # ax1.set_ylim(min_y_ego, max_y_ego)
            # ax1.set_aspect('equal')
            # ax.set_title(f"scene: {scene}, frame: {frame}")
            # # other static_peds_this_frame
            # for _, row in trajectories_ego.iterrows():
            #     ax1.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
            # # ego-agent
            # ax1.add_artist(plt.Circle((0,0), radius=0.5, color='red', fill=True))
            # # ego rotation
            # ax1.add_artist(plt.Arrow(0,0, 10, 0, width=1, color='red'))

            # save fig
            plt.tight_layout()

            if t == items[::-1][0]:
                plt.savefig(f'{ego_traj_save_dir}/{scene}_temp_2d.png')

            fig.canvas.draw()
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(plot_image)
            plt.close(fig)

        fps = 2.5
        output_file_3d = f'/home/eweng/code/viz/jrdb_egomotion_static_opt'
        if not os.path.exists(output_file_3d):
            os.makedirs(output_file_3d)
        output_file_3d = f'{output_file_3d}/{scene}_new_opt.mp4'
        with imageio.get_writer(output_file_3d, fps=fps) as writer:
            for frame in frames[::-1]:
                writer.append_data(frame)
        print(f"saved to: {output_file_3d}")

    # add egomotion in as an additional pedestrian to the df
    ego_ped_id = 1000
    ego_ped_df = pd.DataFrame({'frame': np.arange(len(images_0)), 'id': ego_ped_id, 'x': ego_positions[:, 0], 'y': ego_positions[:, 1], 'heading': rotations})
    df = pd.concat([df, ego_ped_df], ignore_index=True)
    df = df.sort_values(by=['frame', 'id']).reset_index(drop=True)

    # save new trajectories
    if args.save_traj:
        # bev_traj_adjusted_path = f'/home/eweng/code/PoseFormer/datasets/jackrabbot/jrdb_adjusted/{scene}.txt'
        bev_traj_adjusted_path = f'/home/eweng/code/AgentFormerSDD/datasets/jrdb_adjusted/{scene}.txt'
        if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
            os.makedirs(os.path.dirname(bev_traj_adjusted_path))
        print(f"df: {df}")
        df = df[df['frame'] % skip == 0]
        print(f"df: {df}")
        df.to_csv(bev_traj_adjusted_path, index=False, header=False, sep=' ')


if __name__ == "__main__":
    scene_to_list_of_static_ped_ids = {
            'clark-center-2019-02-28_0': {13:114,18: 138, 108:None,110:None,152:None,153:None,155:None,150:None,
                                          162:None,149:None,154:None},
            'nvidia-aud-2019-04-18_0': {20: None, 27: None, 31: None, 37: None},  # small amount of rotation
            'huang-basement-2019-01-25_0': {1: None,  10: None, 11: None, 17: None, #18: None,# 29: None, 6: None,
                                            33: None, 34: None,  45: None, 46: None,}, #53: None }, 40: None, 42: None,
    }
    scene_to_list_of_static_ped_ids1 = {
            # not all scenes have at least 1 static ped
            'clark-center-2019-02-28_1': {204: None, 208: None, 58: (0, 27), 170: None, 150: None},
            'clark-center-intersection-2019-02-28_0': {103: None, 104: None, 105: None, 110: None, 7: 69, 8: 69},
            # 7:117,8:117,},

            # ranges can make it work out
            'forbes-cafe-2019-01-22_0': {11:None,122:None,124:None,3:(1335,1434), 48:(1314, np.inf), 58:None,115:None},
            'memorial-court-2019-03-16_0': {25:None, 52:None, 47:None},
            'huang-2-2019-01-25_0': { 9:[(0,117), (396, 498), (609, np.inf)],
                                      10:[(0,117), (396, 498), (780, np.inf)], 16:None, 18:None, 21:None, 24:None, 4:None},
            'packard-poster-session-2019-03-20_0': {},#{31:300, 41:300},  # some rotation and movement
    }

            # movement restricted to small timeframes
    scene_to_list_of_static_ped_ids2 = {
            'cubberly-auditorium-2019-04-22_0': {0: None, 1: None},  # small amount of rotation
            # movement: (96, 153), (468, 690)
            # 'tressider-2019-04-26_2': {0:[(201, 420), ]}
            # movement: (201, 258), (903, 1032), (1203, end)
    }

    # no static peds, should process by some other method
    '''
    'gates-to-clark-2019-02-28_1': {},  # linear movement
    'gates-159-group-meeting-2019-04-03_0': {},
    'meyer-green-2019-03-16_0': {},  # some rotation and movement
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', '-mp', action='store_true')
    parser.add_argument('--save_traj', '-st', action='store_true')
    parser.add_argument('--dont_visualize', '-dv', dest='visualize', action='store_false')
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
