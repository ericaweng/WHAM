""" Format JRDB 2D body keypoints into AgentFormer-consumable format. """

import joblib
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import yaml

from jrdb_split import TRAIN


COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]

def draw_pose_2d(pose):
    fig, ax = plt.subplots()
    for j1, j2 in COCO_CONNECTIVITIES_LIST:
        x = np.array([pose[j1, 0], pose[j2, 0]])
        y = np.array([pose[j1, 1], pose[j2, 1]])
        ax.plot(x, y, lw=2, color='b')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # plot texts
    for i in range(17):
        ax.text(pose[i, 0], pose[i, 1], str(i), fontsize=8, color='black')
    x_min, x_max = pose[:, 0].min(), pose[:, 0].max()
    y_min, y_max = pose[:, 1].min(), pose[:, 1].max()
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    plt.savefig('../viz/pose_2d.png')


def load_poses(scene_name, args):
    """ detect 3d poses for all detections that have a fully_visible or partially_visible bbox detection
     OR have a 2d pose keypoint annotations """
    # scene_name_w_image = video.split('/')[-1].split(".")[0].replace('image_', 'image')
    scene_name_w_image = scene_name

    # load poses
    with open(os.path.join(args.dataroot_poses, f'{scene_name_w_image}.json'), 'r') as f:
        pose_labels = json.load(f)
    image_id_to_pose_annos = {
            int(image['file_name'].split('/')[-1].split('.')[0]): [ann for ann in pose_labels['annotations']
                                                                   if ann['image_id'] == image['id']]
            for image in pose_labels['images']}

    # convert poses visibility to score percentages
    all_poses = {}
    for image_id, annos in sorted(image_id_to_pose_annos.items(), key=lambda x: x[0]):
        poses_this_frame = {}
        for ann in annos:
            pose_reformatted = np.array(ann['keypoints']).reshape(17, 3).astype(float)
            # negate y-axis
            pose = pose_reformatted[:, :2]
            pose[:, 1] = pose_labels['images'][image_id]['height'] - pose[:, 1]
            # normalize poses by their min and max x and y
            pose[:, 0] = (pose[:, 0] - pose[:, 0].min()) / (pose[:, 0].max() - pose[:, 0].min())
            pose[:, 1] = (pose[:, 1] - pose[:, 1].min()) / (pose[:, 1].max() - pose[:, 1].min())
            # normalize with body root (the midpoint of joint 4 and 8)
            body_root = pose[8]#(pose[4] + pose[8]) / 2
            pose -= body_root

            # convert visibility to confidence score
            score = pose_reformatted[:, -1]
            score = np.where(score == 0, 0.1, score)
            score = np.where(score == 1, 0.5, score)
            score = np.where(score == 2, 1, score)
            assert np.all([np.isclose(p, 0.1) or np.isclose(p, 0.5) or np.isclose(p, 1) for p in score]), f"score: {score}"

            poses_this_frame[ann['track_id']] = {'pose': pose, 'score': score}

        all_poses[image_id] = poses_this_frame

    # video_length, num_peds, {pose: (17, 2), score: (17, 1)}
    return all_poses


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Trajectory Visualization')
    # parser.add_argument('--dataroot_poses', '-dr', type=str, default=f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/")
    parser.add_argument('--dataroot_poses', '-dr', type=str, default=f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_stitched_coco/")
    args = parser.parse_args()

    # 2d poses stitched
    # poses_stitched = {}
    # for scene in TRAIN:
    #     poses_stitched[scene] = load_poses(scene, args)
    # np.savez('../AgentFormerSDD/datasets/jrdb_adjusted/poses_stitched_2d.npz', **poses_stitched)
    # print(f"saved to: ../AgentFormerSDD/datasets/jrdb_adjusted/poses_stitched_2d.npz")
    # import ipdb; ipdb.set_trace()

    args.dataroot_poses = "/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"

    # load camera calibration params for jrdb
    jrdb_calib_path = 'jrdb/train/calibration/cameras.yaml'
    with open(jrdb_calib_path) as f:
        camera_config_dict = yaml.safe_load(f)

    calib_params = {}
    for cam_num in [0, 2, 4, 6, 8]:
        camera_params = camera_config_dict['cameras'][f'sensor_{cam_num}']
        K = camera_params['K'].split(' ')
        fx, fy, cx, cy = K[0], K[2], K[4], K[5]
        calib_params[cam_num] = list(map(float, [fx, fy, cx, cy, *camera_params['D'].split(' ')]))  # intrinsic + distortion

    # merge poses from different cams. if two cams have the same pose, then pick the pose with greater number of visible keypoints
    all_poses = {}
    for cam_num in [0, 2, 4, 6, 8]:
        all_poses[cam_num] = {}
        for scene in TRAIN:
            scene_name = f'{scene}_image{cam_num}'
            all_poses[cam_num][scene] = load_poses(scene_name, args)

    combined_poses = {}
    cam_ids = {}
    scores = {}
    for scene, this_scene_poses in all_poses[0].items():
        combined_poses[scene] = {}
        cam_ids[scene] = {}
        scores[scene] = {}
        for frame, poses in this_scene_poses.items():
            combined_poses[scene][frame] = {}
            cam_ids[scene][frame] = {}
            scores[scene][frame] = {}
            for ped_id, pose in poses.items():
                combined_poses[scene][frame][ped_id] = pose['pose']#{'pose': pose, 'cam_id': 0, 'intrinsics': calib_params[0]}
                cam_ids[scene][frame][ped_id] = 0
                scores[scene][frame][ped_id] = pose['score']

    for cam_num in [2, 4, 6, 8]:
        for scene, this_scene_poses in all_poses[cam_num].items():
            for frame, this_frame_poses in this_scene_poses.items():
                for ped_id, pose in this_frame_poses.items():
                    if (ped_id in combined_poses[scene][frame]
                            and np.sum(pose['score']) > np.sum(scores[scene][frame][ped_id])
                    or ped_id not in combined_poses[scene][frame]):
                        combined_poses[scene][frame][ped_id] = pose['pose']
                        cam_ids[scene][frame][ped_id] = cam_num
                        scores[scene][frame][ped_id] = pose['score']

    print(f"combined_poses: {combined_poses['bytes-cafe-2019-02-07_0'][0][1]}")
    print(f"combined_poses: {scores['bytes-cafe-2019-02-07_0'][0][1]}")
    np.savez('../AgentFormerSDD/datasets/jrdb_adjusted/poses_2d.npz', poses=combined_poses, cam_ids=cam_ids, scores=scores)


if __name__ == '__main__':
    main()
