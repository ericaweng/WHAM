""" Downsample the jrdb no-egomotion BEV trajectories to match the frame rate of for AgentFormer (2.5 Hz). """
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

from jrdb_split import TRAIN, TEST, NO_MOVEMENT


bev_traj_dir = '/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
skip = 1

for scene in NO_MOVEMENT:
    bev_traj_path = f'{bev_traj_dir}/{scene}.txt'
    df = pd.read_csv(bev_traj_path, sep=' ', header=None)
    df.columns = ['frame', 'id', 'x', 'y', 'heading']

    bev_traj_adjusted_path = f'/home/eweng/code/AgentFormerSDD/datasets/jrdb_adjusted/{scene}.txt'
    if not os.path.exists(os.path.dirname(bev_traj_adjusted_path)):
        os.makedirs(os.path.dirname(bev_traj_adjusted_path))
    print(f"df: {len(df)}")
    df = df[df['frame'] % skip == 0]
    print(f"df: {len(df)}")
    print()
    df.to_csv(bev_traj_adjusted_path, index=False, header=False, sep=' ')
