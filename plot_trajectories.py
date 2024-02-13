"""plot jrdb camera-coordinate trajectories in BEV and ego-perspective to identify static pedestrians"""

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
    bev_traj_dir = args.input_traj_dir
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None)  # , usecols=[0, 1, 10, 11])
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    assert len(df['frame'].unique()) == len(images_0), \
        f"len(df['frame'].unique()): {len(df['frame'].unique())}, len(image_list): {len(images_0)}"

    ego_positions = np.zeros((len(images_0), 2))
    ego_rotations = np.zeros(len(images_0))

    ##### plotting #####
    # Determine the min and max positions for scaling across all frames
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()

    # Assign unique colors to each pedestrian ID
    unique_ids = df['id'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_ids)))
    color_dict = dict(zip(unique_ids, colors))

    skip = 3
    frames = []
    items = list(enumerate(zip(ego_positions, df.groupby('frame'))))[::skip]

    ego_traj_save_dir = f'../viz/jrdb_egomotion_static_opt'
    if not os.path.exists(ego_traj_save_dir):
        os.makedirs(ego_traj_save_dir)

    for (t, (ego_pos, (frame, trajectories))) in tqdm(items[::-1], total=len(images_0) // skip):
        fig, ((ax1, ax, ax_im), (ax_im0, ax_im1, ax_im2)) = plt.subplots(2, 3, figsize=(20, 10))

        # plot camera ego image((image, image2, image4, image6, image8),
        ax_im.imshow(cv2.imread(images_0[t])[..., [2, 1, 0]])  # RGB to BGR
        ax_im.set_title(f"cam 0")
        ax_im2.imshow(cv2.imread(images_2[t])[..., [2, 1, 0]])  # RGB to BGR
        ax_im2.set_title(f"cam 2")
        ax_im1.imshow(cv2.imread(images_4[t])[..., [2, 1, 0]])  # RGB to BGR
        ax_im1.set_title(f"cam 4")
        ax_im0.imshow(cv2.imread(images_6[t])[..., [2, 1, 0]])  # RGB to BGR
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
            # plot agent ego_rotations
            ax.arrow(row['x'], row['y'], -np.cos(row['heading']), np.sin(row['heading']), head_width=0.1,
                     head_length=0.1, fc='m', ec='m')

        # ego agent
        ax.add_artist(plt.Circle(ego_pos[:2], radius=0.5, color='red', fill=True))
        # ego path
        ax.plot(ego_positions[:t, 0], ego_positions[:t, 1], color='red')
        # ego rotation
        theta = ego_rotations[t]
        theta_str = str(round(theta, 2)) + " pi"
        ax.add_artist(plt.Arrow(ego_pos[0], ego_pos[1], 5 * np.cos(theta), 5 * np.sin(theta), width=1, color='red'))
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0],
                f"ego-agent pos: {round(ego_pos[0], 1), round(ego_pos[1], 1)}, {theta_str}", fontsize=12,
                color='black')

        # ego perspective
        ax1.imshow(cv2.imread(images_8[t])[..., [2, 1, 0]])  # RGB to BGR
        ax1.set_title(f"cam 8")

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
    output_file_3d = args.output_dir
    if not os.path.exists(output_file_3d):
        os.makedirs(output_file_3d)
    output_file_3d = f'{output_file_3d}/{scene}_new_opt.mp4'
    with imageio.get_writer(output_file_3d, fps=fps) as writer:
        for frame in frames[::-1]:
            writer.append_data(frame)
    print(f"saved to: {output_file_3d}")


if __name__ == "__main__":
    scene_to_list_of_static_ped_ids = {
            'clark-center-2019-02-28_0': {13: 114, 18: 138, 108: None, 110: None, 152: None, 153: None, 155: None,
                                          150: None,
                                          162: None, 149: None, 154: None},
            'nvidia-aud-2019-04-18_0': {20: None, 27: None, 31: None, 37: None},  # small amount of rotation
            'huang-basement-2019-01-25_0': {1: None, 10: None, 11: None, 17: None,  # 18: None,# 29: None, 6: None,
                                            33: None, 34: None, 45: None, 46: None, },
            'clark-center-2019-02-28_1': {204: None, 208: None, 58: (0, 27), 170: None, 150: None},
            'clark-center-intersection-2019-02-28_0': {103: None, 104: None, 105: None, 110: None, 7: 69, 8: 69},
            'forbes-cafe-2019-01-22_0': {11: None, 122: None, 124: None, 3: (1335, 1434), 48: (1314, np.inf), 58: None,
                                         115: None},
            'memorial-court-2019-03-16_0': {25: None, 52: None, 47: None},
            'huang-2-2019-01-25_0': {9: [(0, 117), (396, 498), (609, np.inf)],
                                     10: [(0, 117), (396, 498), (780, np.inf)], 16: None, 18: None, 21: None, 24: None,
                                     4: None},
            'packard-poster-session-2019-03-20_0': {},  # {31:300, 41:300},  # some rotation and movement
            'cubberly-auditorium-2019-04-22_0': {0: None, 1: None},  # small amount of rotation
            'gates-to-clark-2019-02-28_1': {},  # linear movement
            'gates-159-group-meeting-2019-04-03_0': {},
            'meyer-green-2019-03-16_0': {},  # some rotation and movement
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', '-mp', action='store_true')
    parser.add_argument('--input_traj_dir', type=str, default='/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw')
    parser.add_argument('--output_dir', type=str, default='/home/eweng/code/viz/jrdb_trajectories')
    args = parser.parse_args()
    __spec__ = None

    if args.mp:
        import multiprocessing as mp
        mp.set_start_method('spawn')
        list_of_args = [(scene, args) for scene in scene_to_list_of_static_ped_ids.keys()]
        with mp.Pool(min(len(list_of_args), mp.cpu_count())) as p:
            p.starmap(main, list_of_args)

    else:
        for scene in scene_to_list_of_static_ped_ids:
            main(scene, args)
