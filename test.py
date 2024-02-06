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

bev_traj_dir = '/home/eweng/code/AgentFormerSDD/datasets/jrdb/raw'
skip = 3

for scene in no_movement:
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
