
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


def visualize_BEV_trajs(df, df_ego, images_0, images_2, images_4, images_6, images_8,
                        scene, args):
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

    # generate frames
    frames = []
    # items = list(enumerate(df.groupby('frame')))[:args.length:args.skip]
    items = list(enumerate(zip(df.groupby('frame'), df_ego.groupby('frame'))))[::args.skip]

    if not os.path.exists(args.output_viz_dir):
        os.makedirs(args.output_viz_dir)

    for (t, ((frame, trajectories), (frame_ego, trajectories_ego))) in tqdm(items[::-1], total=len(images_0) // args.skip):
        # for (t, (frame, trajectories)) in tqdm(items[::-1], total=len(images_0) // args.skip):
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
        # ax1.imshow(cv2.imread(images_8[t])[..., [2, 1, 0]])  # RGB to BGR
        # ax1.set_title(f"cam 8")

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

            EGO_ID = 1000
            if row['id'] == EGO_ID:  # ego-agent
                # ego agent
                ego_pos = trajectories[trajectories['id'] == EGO_ID][['x', 'y']].values
                assert ego_pos.shape[0] == 1, f"ego_pos.shape: {ego_pos.shape}"
                ego_pos = ego_pos.squeeze()
                ax.add_artist(plt.Circle(ego_pos, radius=0.5, color='red', fill=True))
                # ego path
                xs = df[(df['id'] == EGO_ID) & (df['frame'] <= t)]['x'].values
                ys = df[(df['id'] == EGO_ID) & (df['frame'] <= t)]['y'].values
                theta = df[(df['id'] == EGO_ID) & (df['frame'] == t)]['heading'].values.item()
                ax.plot(xs, ys, color='red')
                # ego rotation
                theta_str = str(np.round(theta, 2)) + " pi"
                ax.add_artist(plt.Arrow(ego_pos[0], ego_pos[1], 5 * np.cos(theta), 5 * np.sin(theta), width=1, color='red'))
                ax.text(ax.get_xlim()[0], ax.get_ylim()[0],
                        f"ego-agent pos: {round(ego_pos[0], 1), round(ego_pos[1], 1)}, {theta_str}", fontsize=12,
                        color='black')

        # egomotion
        ax1.set_xlim(min_x_ego, max_x_ego)
        ax1.set_ylim(min_y_ego, max_y_ego)
        ax1.set_aspect('equal')
        ax.set_title(f"scene: {scene}, frame: {frame}")
        # other static_peds_this_frame
        for _, row in trajectories_ego.iterrows():
            ax1.scatter(row['x'], row['y'], s=10, color=color_dict[row['id']])
        # ego-agent
        ax1.add_artist(plt.Circle((0,0), radius=0.5, color='red', fill=True))
        # ego rotation
        ax1.add_artist(plt.Arrow(0,0, 10, 0, width=1, color='red'))

        # save fig
        plt.tight_layout()

        if t == items[::-1][0][0]:
            plt.savefig(f'{args.output_viz_dir}/{scene}_temp_2d.png')

        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(plot_image)
        plt.close(fig)

    fps = 2.5
    viz_dir = args.output_viz_dir
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    viz_dir = f'{viz_dir}/{scene}.mp4'
    with imageio.get_writer(viz_dir, fps=fps) as writer:
        for frame in frames[::-1]:
            writer.append_data(frame)
    print(f"saved to: {viz_dir}")
