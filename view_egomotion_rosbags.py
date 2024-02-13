""" extract egomotion from rosbag odometry data and save to npy """

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import scipy.interpolate
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion

from jrdb_split import TRAIN, WITH_MOVEMENT

class EgomotionEstimatorWithOrientation:
    def __init__(self, bag_file):
        self.scene = bag_file.split('/')[-1].split('.')[0]
        self.bag = rosbag.Bag(bag_file)
        self.states = []
        self.times = []
        self.gps_times = []

    def process_messages(self):
        imu_datas = 0
        odom_datas = 0
        gps_datas = 0
        for topic, msg, t in self.bag.read_messages(topics=['segway/feedback/ext_imu',
                                                            'segway/feedback/wheel_odometry',
                                                            'segway/feedback/gps/fix_2d']):
            if topic == 'segway/feedback/ext_imu':
                # self.times.append(t.to_sec())
                # self.process_imu_data(msg)
                imu_datas +=1
            elif topic == 'segway/feedback/wheel_odometry':
                self.times.append(t.to_sec())
                self.process_odom_data(msg)
                odom_datas += 1
            elif topic == 'segway/feedback/gps/fix_2d':
                # self.times.append(t.to_sec())
                # self.process_gps_data(msg)
                pass
        print(f"imu_datas: {imu_datas}, odom_datas: {odom_datas}, gps_datas: {gps_datas}")

    def process_gps_data(self, gps_data):
        print(f"GPS Position: ({gps_data.latitude}, {gps_data.longitude}, {gps_data.altitude})")
        self.states.append([gps_data.latitude, gps_data.longitude, 0])

    def process_imu_data(self, imu_data):
        # Extract the yaw (rotation) from the IMU quaternion
        x = imu_data.orientation.x
        y = imu_data.orientation.y
        z = imu_data.orientation.z
        orientation_q = Quaternion(imu_data.orientation.w, imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z)
        euler = orientation_q.yaw_pitch_roll
        yaw = euler[0]
        print(f"IMU position + orientation: ({x}, {y}, {z}, {orientation_q})")

        # Update the filter with the new orientation
        self.states.append([x, y, yaw])

    def process_odom_data(self, odom_data):
        # Assuming odometry provides displacement in robot's local frame
        x = odom_data.pose.pose.position.x  # Displacement in x
        y = odom_data.pose.pose.position.y  # Displacement in y
        theta = Quaternion(odom_data.pose.pose.orientation.w, odom_data.pose.pose.orientation.x,
                            odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z).yaw_pitch_roll[0]

        self.states.append([x, y, theta])
        print(f"Odometry Position: ({x}, {y}, {theta})")

    def run(self):
        self.process_messages()
        self.bag.close()


def main(scene, args):
    bag_file = f'/mnt/nas/erica/datasets/jackrabbot/rosbags/train/{scene}.bag'
    estimator = EgomotionEstimatorWithOrientation(bag_file)
    estimator.run()

    # rotate entire trajectory such that initial orientation is 0 and initial position is (0, 0)
    estimator.states = np.array(estimator.states)
    estimator.states[:, :2] -= estimator.states[0, :2]
    yaw = estimator.states[0, 2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    estimator.states[:, :2] = np.dot(estimator.states[:, :2], R)
    estimator.states[:, 2] -= estimator.states[0, 2] + 2 * np.pi

    # downsample to 7.5 Hz. first calculate skip based on the timestamps
    timestamps = np.array(estimator.times)
    time_diffs = np.diff(timestamps)
    print(f"time_diffs: {time_diffs}")

    # number of egomotion samples to take from the odometry data
    num_to_sample = len(sorted(glob.glob(f"jrdb/train/images/image_0/{scene}/*")))
    print(f"num_to_sample: {num_to_sample}")

    f = scipy.interpolate.interp1d(timestamps, estimator.states, axis=0)
    timestamps = np.linspace(timestamps[0], timestamps[-1], num_to_sample)
    timestamps = np.arange(timestamps[0], timestamps[-1], 1/7.5)
    sampled_states = f(timestamps)

    # make yaw between 0 and 2*pi
    sampled_states[:, 2] = sampled_states[:, 2] % (2 * np.pi)
    print(f"sampled_states: {sampled_states.shape}")

    # plot the rotated ego-trajectory
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(sampled_states[:, 0], sampled_states[:, 1], sampled_states[:, 2], label='Ego-motion trajectory')
    fig, ax = plt.subplots()
    ax.plot(sampled_states[:, 0], sampled_states[:, 1])
    for state in sampled_states[::10]:
        ax.add_artist(plt.Arrow(state[0], state[1], np.cos(state[2]), np.sin(state[2]), width=0.5, color='r'))
    ax.set_aspect('equal')

    viz_save_dir = args.viz_save_dir
    if not os.path.exists(viz_save_dir):
        os.makedirs(viz_save_dir)
    plt.savefig(f'{viz_save_dir}/{estimator.scene}.png')

    # save to npy
    egomotion_savedir = args.egomotion_savedir
    if not os.path.exists(egomotion_savedir):
        os.makedirs(egomotion_savedir)

    estimator.states = np.array(estimator.states).squeeze()
    np.save(f'{egomotion_savedir}/{estimator.scene}.npy', sampled_states)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--egomotion_savedir', '-es', type=str, default=f"'jrdb/rosbag_egomotion_imu'")
    parser.add_argument('--viz_save_dir', '-vs', type=str, default=f'../viz/rosbag_imu')
    args = parser.parse_args()

    for scene in WITH_MOVEMENT:
        main(scene, args)
