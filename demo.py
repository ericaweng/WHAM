""" option to use ground truth 2d poses instead of detection, tracking, and 2d pose estimation"""

import os
import json
import yaml
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

from jrdb_split import TRAIN, TEST

try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
except:
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False


def load_bboxes_and_poses(video, length, args):
    """ detect 3d poses for all detections that have a fully_visible or partially_visible bbox detection
     OR have a 2d pose keypoint annotations """
    scene_name_w_image = video.split('/')[-1].split(".")[0].replace('image_', 'image')

    # load poses
    dataroot_poses = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"
    with open(os.path.join(dataroot_poses, f'{scene_name_w_image}.json'), 'r') as f:
        pose_labels = json.load(f)
    image_id_to_pose_annos = {
            int(image['file_name'].split('/')[-1].split('.')[0]): [ann for ann in pose_labels['annotations'] if ann['image_id'] == image['id']]
            # image['id']: [ann for ann in pose_labels['annotations'] if ann['image_id'] == image['id']]
            # image['file_name'].split('/')[-1]: [ann for ann in pose_labels['annotations'] if ann['image_id'] == image['id']]
            for image in pose_labels['images']}

    # image_path_to_id = {image['file_name'].split('/')[-1]: image['id'] for image in pose_labels['images']}
    # image_path_to_id = {image['file_name'].split('/')[-1]: image['id'] for image in pose_labels['images']}

    # load bboxes
    dataroot_labels = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d/"
    bboxes_path = os.path.join(dataroot_labels, f'{scene_name_w_image}.json')
    with open(bboxes_path, 'r') as f:
        bbox_labels = json.load(f)['labels']
    all_bboxes = {int(image_path.split('.')[0]): {int(label['label_id'].split(":")[-1]): label
                   for label in labels}  # if 'visible' in label['attributes']['occlusion']}
                  for image_path, labels in sorted(bbox_labels.items())}
    # all_bboxes = {image_path_to_id[image_path]: {int(label['label_id'].split(":")[-1]): label
    # all_bboxes = {image_path: {int(label['label_id'].split(":")[-1]): label

    assert len(
            all_bboxes) == length, f"there should be a list of bboxes for each video frame. but len(all_bboxes): {len(all_bboxes)}, video length: {length}"


    for image_id in image_id_to_pose_annos:
        assert image_id in all_bboxes, f"all images with poses should have a bbox detection, but image_id {image_id} is not have a bbox. only these image_ids have bboxes: {all_bboxes.keys()}"


    # convert poses visibility to score percentages
    all_poses_and_bboxes = {}
    for image_id, annos in sorted(image_id_to_pose_annos.items(), key=lambda x: x[0]):
        poses_this_frame = {}
        for ann in annos:
            pose_reformatted = np.array(ann['keypoints']).reshape(17, 3)
            pose_reformatted[:, -1] = np.where(pose_reformatted[:, -1] == 0, 1, 1)  # 0.01, 1)
            poses_this_frame[ann['track_id']] = {'pose': pose_reformatted}
            # assert ann['track_id'] in all_bboxes[image_id], f"all poses should have a bboxes detection, but pose track_id {ann['track_id']} is not have a bbox. only these track_ids have bboxes: {all_bboxes[image_id].keys()}"

        for track_id, box in all_bboxes[image_id].items():
            # if 'visible' in box['annotation']['occlusion']:
            #     all_bboxes.append({'bbox': box['box'], 'id': track_id})
            if track_id in poses_this_frame:
                poses_this_frame[track_id]['box'] = box_to_2d_corners(box['box'])
            elif 'attributes' not in box or 'occlusion' not in box['attributes'] or 'visible' in box['attributes']['occlusion']:
                poses_this_frame[track_id] = {'box': box_to_2d_corners(box['box'])}
            elif args.use_all_gt_boxes:
                poses_this_frame[track_id] = {'box': box_to_2d_corners(box['box'])}

        # add boxes to those that do not have one
        for track_id, pose_bbox_dict in poses_this_frame.items():
            if 'box' not in pose_bbox_dict:
                ## get max and min of the keypoints as the box
                poses_this_frame[track_id]['box'] = np.concatenate([np.min(pose_bbox_dict['pose'], axis=0)[:2],
                                                                    np.max(pose_bbox_dict['pose'], axis=0)[:2]])
            x0,y0,x1,y1 = pose_bbox_dict['box']
            assert x1 > x0 and y1 > y0, f"invalid box: {pose_bbox_dict['box']}"

        # all_poses_and_bboxes.append(poses_this_frame)
        all_poses_and_bboxes[image_id] = poses_this_frame

    # video_length, num_objects, {pose: (17, 3), box: (4,)}
    return all_poses_and_bboxes


def load_bboxes_and_poses_old(video, length):
    # load bboxes
    scene_name_w_image = video.split('/')[-1].split(".")[0].replace('image_', 'image')
    dataroot_labels = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d/"
    labels_path = os.path.join(dataroot_labels, f'{scene_name_w_image}.json')
    with open(labels_path, 'r') as f:
        labels = json.load(f)['labels']
    all_bboxes = [{int(label['label_id'].split(":")[-1]): label['box']
                   for label in labels }#if 'visible' in label['attributes']['occlusion']}
                  for _, labels in sorted(labels.items())]
    assert len(
        all_bboxes) == length, f"there should be a list of bboxes for each video frame. but len(all_bboxes): {len(all_bboxes)}, video length: {length}"

    # load poses
    dataroot_poses = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/train/labels/labels_2d_pose_coco/"
    # need to check that pose format is same for wham and coco
    poses_frames = json.load(open(os.path.join(dataroot_poses, f'{scene_name_w_image}.json'), 'r'))
    # CONNECTIVITIES_LIST = poses_frames['categories'][0]['skeleton']
    image_id_to_pose_annos = {
            image['id']: [ann for ann in poses_frames['annotations'] if ann['image_id'] == image['id']]
            for image in poses_frames['images']}

    # convert poses visibility to score percentages
    all_poses = []
    for _, annos in sorted(image_id_to_pose_annos.items(), key=lambda x: x[0]):
        poses_this_frame = {}
        for ann in annos:
            pose_reformatted = np.array(ann['keypoints']).reshape(17, 3)
            pose_reformatted[:, -1] = np.where(pose_reformatted[:, -1] == 0, 1, 1)  # 0.01, 1)
            poses_this_frame[ann['track_id']] = pose_reformatted
        all_poses.append(poses_this_frame)
    # video_length, num_objects, 17, 3 (x, y, score)

    assert len(
        all_poses) == length, f"there should be a list of poses for each video frame. but len(all_poses): {len(all_poses)}, video length: {length}"

    for frame_id, (bboxes, poses) in enumerate(zip(all_bboxes, all_poses)):
        for track_id, pose in poses.items():
            assert track_id in bboxes, f"all poses should have a bboxes detection, but pose track_id {track_id} is not have a bbox. only these track_ids have bboxes: {bboxes.keys()}"

    return all_bboxes, all_poses


def box_to_2d_corners(box):
    x, y, w, h = box
    x0, x1 = x, x + w
    y0, y1 = y, y + h
    return np.array([x0, y0, x1, y1])

# COCO 2d / 3d
CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12], [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]

def show_image(image, bboxes=None, poses_gt=None, poses_est=None):
    """
    bboxes: only the gt
    poses_gt: only the gt
    poses_est: only the estimated poses
    """
    if bboxes is not None:
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    if poses_gt is not None:
        draw_pose_2d_single_frame(poses_gt, image, color=(0, 0, 255), connectivities=CONNECTIVITIES_LIST)
    if poses_est is not None:
        draw_pose_2d_single_frame(poses_est, image, color=(255, 0, 0), connectivities=CONNECTIVITIES_LIST)
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def draw_pose_2d_single_frame(pose, image, color, text=False, connectivities=None):
    """
    Draw a single pose for a given frame index.
    pose: (num_peds, num_joints, 3)
    """
    connectivity = connectivities if connectivities is not None else CONNECTIVITY_DICT
    for ped_i in range(pose.shape[0]):  # for each ped
        vals = pose[ped_i]
        for j1, j2 in connectivity:
            image = cv2.line(image, vals[j1].round().astype(np.int32), vals[j2].round().astype(np.int32), color, 2)

        # label joints with index of joint
        if text:
            for i in range(vals.shape[0]):
                cv2.putText(image, str(i), vals[i])


def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        visualize=False):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video length: {length}")
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global

    #### PART 1: Preprocess (tracking, 2d pose, SLAM)
    if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and
            osp.exists(osp.join(output_pth, 'slam_results.pth'))):

        detector = DetectionModel(cfg.DEVICE.lower())
        extractor = FeatureExtractor(cfg.DEVICE.lower())

        if run_global:
            slam = SLAMModel(video, output_pth, width, height, calib)
        else:
            slam = None

        if args.use_gt_poses and video.split('/')[-1].split("_image")[0] in TRAIN:
            all_poses_and_bboxes = load_bboxes_and_poses(video, length, args)

        bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
        while (cap.isOpened()):
            current_frame = bar.index
            flag, img = cap.read()
            if not flag: break

            # for each bounding box: if not gt pose exists, then do pose
            if args.use_gt_poses and video.split('/')[-1].split("_image")[0] in TRAIN and current_frame in all_poses_and_bboxes:
                # for each ped id for which there is a box, populate the tracker with its pose if it exists.
                # if no pose exists for a gt box, detect it.

                # video_length, num_objects, {pose: (17, 3), box: (4,)}
                bboxes_to_est_pose = []
                for track_id, bbox_pose_dict in all_poses_and_bboxes[current_frame].items():
                    if 'pose' in bbox_pose_dict:
                        # load gt 2d tracks into tracking_results
                        detector.tracking_results['id'].append(track_id)
                        detector.tracking_results['frame_id'].append(detector.frame_id)  # could also use current_frame
                        assert detector.frame_id == current_frame, f"should be the same frame_id. but detector.frame_id: {detector.frame_id}, current_frame: {current_frame}"
                        detector.tracking_results['bbox'].append(detector.xyxy_to_cxcys(bbox_pose_dict['box']))
                        detector.tracking_results['keypoints'].append(bbox_pose_dict['pose'][None])

                    else:  # no gt pose exists for this bbox. so detect it using ViTPose
                        bboxes_to_est_pose.append({'bbox': bbox_pose_dict['box'], 'id': track_id})

                if len(bboxes_to_est_pose) > 0:
                    assert (len(bboxes_to_est_pose) + np.sum(['pose' in bbox_pose_dict for bbox_pose_dict in all_poses_and_bboxes[current_frame].values()])
                            == len(all_poses_and_bboxes[current_frame])), \
                        f"should have a pose for each bbox. but len(bboxes): {len(bboxes_to_est_pose)}, len(all_poses_and_bboxes[current_frame]): {len(all_poses_and_bboxes[current_frame])}"
                    detector.pose_estimation(img, bboxes_to_est_pose)  # logging the tracking results already occurs in the function
                    assert np.all([d.shape == detector.tracking_results['bbox'][0].shape for d in detector.tracking_results['bbox']]), \
                        f"should have the same shape. but [d.shape for d in detector.tracking_results['bbox']]: {[d.shape for d in detector.tracking_results['bbox']]}"
                    assert np.all([d.shape == detector.tracking_results['keypoints'][0].shape for d in detector.tracking_results['keypoints']]), \
                        f"should have the same shape. but [d.shape for d in detector.tracking_results['keypoints']]: {[d.shape for d in detector.tracking_results['keypoints']]}"
                else:
                    detector.frame_id += 1

            else:
                # 2D detection and tracking from scratch w/o gt bboxes nor poses
                # detector.track(img, fps, length)
                detector.frame_id += 1

            if args.visualize:
                pass
                # if args.use_gt_poses:
                #     estimated_poses = np.concatenate(detector.tracking_results['keypoints'][-len(bboxes_to_est_pose):])[..., :2]
                #     gt_poses = np.array([d['pose'] for d in all_poses_and_bboxes[current_frame].values()])[...,:2]
                #     gt_bboxes = [box['bbox'] for box in bboxes]
                #     show_image(img, gt_bboxes, gt_poses, estimated_poses)

            # SLAM
            if slam is not None:
                slam.track()

            bar.next()

        tracking_results = detector.process(fps)

        if slam is not None:
            slam_results = slam.process()
        else:
            slam_results = np.zeros((length, 7))
            slam_results[:, 3] = 1.0  # Unit quaternion

        # Extract image features
        # TODO: Merge this into the previous while loop with an online bbox smoothing.
        tracking_results = extractor.run(video, tracking_results)
        logger.info('Complete Data preprocessing!')

        # Save the processed data
        joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
        joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Save processed data at {output_pth}')

    # If the processed data already exists, load the processed data
    else:
        tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
        slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Already processed data exists at {output_pth} ! Load the data .')

    #### PART 2: 3d pose lifting and move poses from camera coords to world coords
    if not osp.exists(osp.join(output_pth, "results.pth")):
        # Build dataset
        dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

        # run WHAM
        results = defaultdict(dict)

        for batch in dataset:
            if batch is None: break

            # data
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch

            # inference
            pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True,
                           **kwargs)

            # Store results
            results[_id]['poses_body'] = pred['poses_body'].detach().cpu().squeeze(0).numpy()
            results[_id]['poses_root_cam'] = pred['poses_root_cam'].detach().cpu().squeeze(0).numpy()
            results[_id]['betas'] = pred['betas'].detach().cpu().squeeze(0).numpy()
            results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).detach().cpu().numpy()
            results[_id]['poses_root_world'] = pred['poses_root_world'].detach().cpu().squeeze(0).numpy()
            results[_id]['trans_world'] = pred['trans_world'].detach().cpu().squeeze(0).numpy()
            results[_id]['frame_id'] = frame_id

        # save results
        joblib.dump(results, osp.join(output_pth, 'results.pth'))
        logger.info(f'Save results at {output_pth}')

    # Visualize
    if visualize and not osp.exists(osp.join(output_pth, 'output.mp4')):
        results = joblib.load(osp.join(output_pth, 'results.pth'))
        from lib.vis.run_vis import run_vis_on_demo
        run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='examples/demo_video.mp4',
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='../viz/wham-demo',
                        help='output folder to write results')

    parser.add_argument('--calib', type=str, default=None,
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', '-lo', action='store_true',
                        help='Only estimate motion in camera coordinate if True')

    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize the output mesh if True')

    parser.add_argument('--dont_use_gt_poses', dest='use_gt_poses', action='store_false')#true')
    parser.add_argument('--use_all_gt_boxes', action='store_true')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    # input_vid_dir = "../viz/wham_input_vids"
    # scenes = os.listdir(input_vid_dir)

    # TRAIN only
    with_movement = ['clark-center-2019-02-28_0',
                     'clark-center-2019-02-28_1',
                     'clark-center-intersection-2019-02-28_0',
                     'cubberly-auditorium-2019-04-22_0',  # small amount of rotation
                     'forbes-cafe-2019-01-22_0',
                     'gates-159-group-meeting-2019-04-03_0',
                     'gates-to-clark-2019-02-28_1',  # linear movement
                     'memorial-court-2019-03-16_0',
                     'huang-2-2019-01-25_0',
                     'huang-basement-2019-01-25_0',
                     'meyer-green-2019-03-16_0',  # some rotation and movement
                     'nvidia-aud-2019-04-18_0',  # small amount of rotation
                     'packard-poster-session-2019-03-20_0',  # some rotation and movement
                     'tressider-2019-04-26_2',]

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

    for camera_num in range(0, 9, 2):
        for scene in no_movement:#with_movement:
            # if 'clark-center-2' not in scene:
            # if 'clark-center-i' not in scene and 'forbes' not in scene:
            # if 'cubberly' not in scene and 'memorial' not in scene:
            # if 'gates' not in scene:
            # if 'huang' not in scene:
            # if 'meyer' not in scene and 'nvidia' not in scene:
            # if 'packard' not in scene and 'tressider' not in scene:
            #     continue
            print(f"scene: {scene}")
            scene = f"{scene}_image_{camera_num}.mp4"
            input_vid_dir = "../viz/wham_input_vids"
            args.video = f"{input_vid_dir}/{scene}"

            logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
            logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

            # ========= Load WHAM ========= #
            smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
            smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
            network = build_network(cfg, smpl)
            network.eval()

            # Output folder
            sequence = args.video.split('/')[-1].split('.')[0]  # '.'.join(args.video.split('/')[-1].split('.')[:-1])
            output_pth = osp.join(args.output_pth, sequence)
            os.makedirs(output_pth, exist_ok=True)

            # load camera calibration params
            jrdb_calib_path = 'jrdb/train/calibration/cameras.yaml'
            camera_num = int(scene.split("_")[-1].split('.')[0])

            with open(jrdb_calib_path) as f:
                camera_config_dict = yaml.safe_load(f)

            camera_params = camera_config_dict['cameras'][f'sensor_{camera_num}']
            K = camera_params['K'].split(' ')
            fx, fy, cx, cy = K[0], K[2], K[4], K[5]
            calib_params = list(map(float, [fx, fy, cx, cy, *camera_params['D'].split(' ')]))  # intrinsic + distortion

            args.calib = osp.join(output_pth, 'calib.txt')
            if not osp.exists(args.calib):
                with open(args.calib, 'w') as fopen:
                    print(" ".join(map(str, calib_params)))
                    fopen.write(" ".join(map(str, calib_params)))

            with torch.no_grad():
                run(cfg,
                    args.video,
                    output_pth,
                    network,
                    args.calib,
                    run_global=not args.estimate_local_only,
                    visualize=args.visualize)

            print()
            logger.info('Done !')