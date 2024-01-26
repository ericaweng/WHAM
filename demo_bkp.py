""" option to use gt bboxes instead of detection"""

import os
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

try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
except:
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False


def box_to_2d_corners(box):
    x, y, w, h = box
    x0, x1 = x, x + w
    y0, y1 = y, y + h
    return [x0, y0, x1, y1]


def show_image(image, bboxes=None):
    if bboxes is not None:
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


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

        if args.use_gt_bboxes:
            import json
            args.split = 'train'
            # dataroot_labels = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/{args.split}/labels/labels_2d_stitched/"
            dataroot_labels = f"/home/eweng/code/PoseFormer/datasets/jackrabbot/{args.split}/labels/labels_2d/"
            scene = args.video.split('/')[-1].split('.')[0]
            scene = scene.replace('image_', 'image')
            labels = json.load(open(os.path.join(dataroot_labels, f'{scene}.json'), 'r'))['labels']
            all_bboxes = [[label['box'] for label in labels] for _, labels in sorted(labels.items())]
            assert len(all_bboxes) == length, f"len(all_bboxes): {len(all_bboxes)}, video length: {length}"

        bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
        current_frame = 0

        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag: break

            if args.use_gt_bboxes:
                bboxes = [box_to_2d_corners(box) for box in all_bboxes[current_frame]]
            else:
                bboxes = None

            current_frame += 1

            # 2D detection and tracking
            detector.track(img, fps, length, bboxes)

            # if args.visualize:
            #     show_image(img, bboxes)

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

    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)

    # save results
    joblib.dump(results, osp.join(output_pth, 'results.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='examples/demo_video.mp4',
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='../pose_forecasting/viz/wham-demo',
                        help='output folder to write results')

    parser.add_argument('--calib', type=str, default=None,
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')

    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')

    parser.add_argument('--use_gt_bboxes', action='store_true')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    input_vid_dir = "../pose_forecasting/viz/wham_input_vids"
    scenes = os.listdir(input_vid_dir)
    for scene in scenes:
        args.video = f"{input_vid_dir}/{scene}"
        print(f"scene: {scene}")

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

        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and
                osp.exists(osp.join(output_pth, 'slam_results.pth')) and
                osp.exists(osp.join(output_pth, 'output.mp4'))):

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