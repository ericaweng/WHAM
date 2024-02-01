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

def get_mask(groupby_data, static_ped_ids):
    # Concatenate all dataframes to find unique pedestrian IDs
    all_frames = pd.concat([group[1] for group in groupby_data], ignore_index=True)
    unique_ids = all_frames['id'].unique()
    num_frames = len(groupby_data)
    num_pedestrians = len(unique_ids)

    # Initialize the array and mask
    static_pedestrians = np.zeros((num_frames, num_pedestrians, 2))
    presence_mask = np.zeros((num_frames, num_pedestrians))

    # Fill in the data
    for frame, df in groupby_data:
        for index, row in df.iterrows():
            ped_index = np.where(unique_ids == row['id'])[0][
                0]  # Find the index of the pedestrian ID in the unique IDs array
            if frame < static_ped_ids[row['id']] or static_ped_ids[row['id']] is None:
                static_pedestrians[frame, ped_index] = [row['x'], row['y']]
                presence_mask[frame, ped_index] = 1

    return static_pedestrians, presence_mask


def optimize_rot_trans_all_ts(static_pedestrians, presence_mask, target, timesteps, initial_guess=None):
    """
    Optimize the rotation angle and delta trajectory over all timesteps, considering the presence or absence of static pedestrians at each timestep.

    :param static_pedestrians: Array of shape (n_static, 2) containing the initial positions of static pedestrians.
    :param target: Array of shape (n_static, 2) containing the target positions (initial positions) of the static pedestrians.
    :param timesteps: Number of timesteps to consider for the delta trajectory.
    :param presence_mask: Array of shape (timesteps, n_static) indicating the presence (1) or absence (0) of each pedestrian at each timestep.
    :param initial_guess: Initial guess for the optimization parameters. If None, defaults to zeros.
    :return: Optimized rotation angle and delta trajectories (dx, dy) over timesteps.
    """
    if initial_guess is None:
        initial_guess = np.zeros(timesteps * 3)  # 3 parameters per timestep: rotation, delta_x, delta_y

    def cost_function(params):
        rotations = params[::3]
        delta_xs = params[1::3]
        delta_ys = params[2::3]

        cos_rotations = np.cos(rotations)
        sin_rotations = np.sin(rotations)
        rotation_matrices = np.stack((cos_rotations, -sin_rotations, sin_rotations, cos_rotations), axis=1).reshape(-1, 2, 2)

        rotated_positions = np.einsum('tni,tij->tnj', static_pedestrians, rotation_matrices)
        final_positions = rotated_positions + np.stack((delta_xs, delta_ys), axis=1)[:, np.newaxis, :]

        # Apply the presence mask to include only the cost for present pedestrians
        masked_cost = np.sum(presence_mask[:, :, np.newaxis] * (final_positions - target[np.newaxis]) ** 2)

        delta_x_diffs = np.diff(delta_xs)
        delta_y_diffs = np.diff(delta_ys)
        penalty = 0.1 * np.sum(delta_x_diffs ** 2 + delta_y_diffs ** 2)

        return masked_cost + penalty

    result = minimize(cost_function, initial_guess.flatten(), method='Nelder-Mead')
    optimized_params = result.x.reshape((timesteps, 3))
    optimized_rotation = optimized_params[:, 0]
    optimized_delta = optimized_params[:, 1:]

    return optimized_rotation, optimized_delta


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


def main(scene, args):
    # Example usage (Note: Replace these with your actual data)
    print(f"scene: {scene}")
    split = 'train' if scene in TRAIN else 'test'

    # load in ego-perspective camera rgb images for plotting
    images_0 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_0/{scene}/*"))]
    images_2 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_2/{scene}/*"))]
    images_4 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_4/{scene}/*"))]
    images_6 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_6/{scene}/*"))]
    images_8 = [cv2.imread(imfile) for imfile in sorted(glob.glob(f"jrdb/{split}/images/image_8/{scene}/*"))]

    # load BEV 2d trajs
    bev_traj_dir = '/home/eweng/code/PoseFormer/try2/CIWT/data'
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None, usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    df_ego = df.copy()

    # calculate egomotion for each timestep
    static_ped_ids = scene_to_list_of_static_ped_ids[scene]
    init_static_poses_df = df[df['id'].isin(static_ped_ids)]
    init_static_poses = init_static_poses_df[init_static_poses_df['frame'] == 0]
    static_pedestrians = init_static_poses_df.groupby('frame')
    assert len(static_pedestrians) == len(images_0), \
        f'len(static_pedestrians): {len(static_pedestrians)}, len(image_list): {len(images_0)}'
    print(f"init_static_poses: {init_static_poses.shape} should be (n_static, 2)")

    if args.joint_opt:
        # load in initial guess
        # traj_path = f"../viz/wham-demo/{scene}_image_0/slam_results.pth"
        # try:
        #     initial_guess = joblib.load(traj_path)
        #     initial_guess = initial_guess[..., [2, 0, 1]]
        #     initial_guess[..., 1] = -initial_guess[..., 1]  # negate prev-x-now-y
        #     initial_guess[..., 2] = 0
        # except:
        #     print(f"no dvpo results at {traj_path} ({scene} image{cam_num})")
        initial_guess = None

        static_pedestrians_arr, mask = get_mask(static_pedestrians, static_ped_ids)
        # calculate egomotion transformation per timestep
        rotations, ego_positions = optimize_rot_trans_all_ts(static_pedestrians_arr, mask, static_pedestrians_arr[0], len(images_0), initial_guess)
        delta_x, delta_y = ego_positions[:, 0], ego_positions[:, 1]
        assert np.isclose(rotations[0], 0, atol=1e-4), f"rotation: {rotations}"
        assert np.allclose(ego_positions[0], 0, atol=1e-4), f"ego_positions[0] should be close to 0: {ego_positions[0]}"

    else:
        rotations = []
        ego_positions = []
        for t, peds in static_pedestrians:
            joined = peds.merge(init_static_poses, on='id', suffixes=('','_init'), how='inner')
            curr_ts_poses = joined[['x', 'y']].values
            init_poses = joined[['x_init', 'y_init']].values
            # print(f"len(peds): {len(peds)}")
            rotation, delta_x, delta_y = (
                    optimize_rotation_and_delta(curr_ts_poses, init_poses))
            if t == 0:
                assert np.isclose(rotation, 0), f"rotation: {rotation}"
                assert np.isclose(delta_x, 0), f"delta_x: {delta_x}"
                assert np.isclose(delta_y, 0), f"delta_y: {delta_y}"
            rotations.append(rotation)
            ego_positions.append([delta_x, delta_y])
        ego_positions = smooth_trajectory(np.array(ego_positions), sigma=5)
        delta_x, delta_y = ego_positions[:, 0], ego_positions[:, 1]
        rotations = smooth_rotations(rotations, sigma=1)

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

    frames = []
    skip = 3
    items = list(zip(images_0, ego_positions, df.groupby('frame'), df_ego.groupby('frame')))[::skip]

    ego_traj_save_dir = f'../viz/jrdb_egomotion_static_opt'
    if not os.path.exists(ego_traj_save_dir):
        os.makedirs(ego_traj_save_dir)

    for (t, (image, ego_pos, (frame, trajectories), (frame_ego, trajectories_ego))) in tqdm(list(enumerate(items))[::-1], total=len(images_0) // skip):
        fig, ((ax1, ax, ax_im),(ax_im0,ax_im1,ax_im2)) = plt.subplots(2, 3, figsize=(20, 10))
        assert frame == frame_ego, f"frame: {frame}, frame_ego: {frame_ego}"

        # plot camera ego image
        ax_im.imshow(image[...,[2,1,0]])  # RGB to BGR
        ax_im.set_title(f"cam 0")
        ax_im0.imshow(images_2[t,...,[2,1,0]])  # RGB to BGR
        ax_im0.set_title(f"cam 2")
        ax_im1.imshow(images_4[t,...,[2,1,0]])  # RGB to BGR
        ax_im1.set_title(f"cam 4")
        ax_im2.imshow(images_6[t,...,[2,1,0]])  # RGB to BGR
        ax_im2.set_title(f"cam 6")

        # global perspective
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_title(f"scene: {scene}, frame: {frame}")
        # other peds
        for _, row in trajectories.iterrows():
            ax.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
            ax.annotate(int(row['id']), (row['x'], row['y']), fontsize=6)
        # ego agent
        ax.add_artist(plt.Circle(ego_pos[:2], radius=0.5, color='red', fill=True))
        # ego path
        ax.plot(ego_positions[:t*skip:skip, 0], ego_positions[:t*skip:skip, 1], color='red')
        # ego rotation
        rot_mat = frame_to_ego_rot_mat[frame]
        theta = np.arctan2(-rot_mat[1, 0], rot_mat[0, 0]) / np.pi
        theta = str(round(theta, 2)) + " pi"
        ax.add_artist(plt.Arrow(ego_pos[0], ego_pos[1], 5*rot_mat[0,0], -5*rot_mat[1,0], width=1, color='red'))
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
        plt.tight_layout()

        if t == len(images_0) // skip - 1:
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
    output_file_3d = f'{output_file_3d}/{scene}.mp4'
    with imageio.get_writer(output_file_3d, fps=fps) as writer:
        for frame in frames[::-1]:
            writer.append_data(frame)
    print(f"saved to: {output_file_3d}")

    # save new trajectories
    bev_traj_adjusted_path = f'/home/eweng/code/PoseFormer/datasets/jackrabbot/jrdb_adjusted/{scene}.txt'
    if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
        os.makedirs(os.path.dirname(bev_traj_adjusted_path))
    print(f"df: {df}")
    df = df[df['frame'] % skip == 0]
    print(f"df: {df}")
    df.to_csv(bev_traj_adjusted_path, index=False)


if __name__ == "__main__":
    scene = 'clark-center-2019-02-28_0'  # 'bytes-cafe-2019-02-07_0'##'tressider-2019-04-26_2'#
    scene_to_list_of_static_ped_ids = {
            'clark-center-2019-02-28_0': {13:114,18: 138, 108:None,110:None,152:None,153:None,155:None,150:None,162:None,149:None,154:None},
            'clark-center-2019-02-28_1': {},
            'clark-center-intersection-2019-02-28_0': {},
            'cubberly-auditorium-2019-04-22_0': {},  # small amount of rotation
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
    # logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_opt', action='store_true')
    args = parser.parse_args()
    main(scene, args)