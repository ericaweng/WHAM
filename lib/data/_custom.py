from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from configs import constants as _C
from .normalizer import Normalizer
from lib.utils import transforms
from lib.models import build_body_model
from lib.utils.kp_utils import root_centering
from lib.utils.imutils import compute_cam_intrinsics

KEYPOINTS_THR = 0.3

def convert_dpvo_to_cam_angvel(traj, fps):
    """Function to convert DPVO trajectory output to camera angular velocity"""
    
    # 0 ~ 3: translation, 3 ~ 7: Quaternion
    quat = traj[:, 3:]
    
    # Convert (x,y,z,q) to (q,x,y,z)
    quat = quat[:, [3, 0, 1, 2]]
    
    # Quat is camera to world transformation. Convert it to world to camera
    world2cam = transforms.quaternion_to_matrix(torch.from_numpy(quat)).float()
    R = world2cam.mT
    
    # Compute the rotational changes over time.
    cam_angvel = transforms.matrix_to_axis_angle(R[:-1] @ R[1:].transpose(-1, -2))
    
    # Convert matrix to 6D representation
    cam_angvel = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(cam_angvel))
    
    # Normalize 6D angular velocity
    cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel) # Normalize
    cam_angvel = cam_angvel * fps
    cam_angvel = torch.cat((cam_angvel, cam_angvel[:1]), dim=0)
    return cam_angvel


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tracking_results, slam_results, width, height, fps):
        
        self.tracking_results = tracking_results
        self.slam_results = slam_results
        self.width = width
        self.height = height
        self.fps = fps
        self.res = torch.tensor([width, height]).float()
        self.intrinsics = compute_cam_intrinsics(self.res)
        
        self.device = cfg.DEVICE.lower()
        
        self.smpl = build_body_model('cpu')
        self.keypoints_normalizer = Normalizer(cfg)
        
        self._to = lambda x: x.unsqueeze(0).to(self.device)
        
    def __len__(self):
        return len(self.tracking_results.keys())
    
    def __getitem__(self, _index):
        if _index >= len(self): return
        index = sorted(list(self.tracking_results.keys()))[_index]
            
        # Process 2D keypoints
        kp2d = torch.from_numpy(self.tracking_results[index]['keypoints']).float()
        mask = kp2d[..., -1] < KEYPOINTS_THR
        bbox = torch.from_numpy(self.tracking_results[index]['bbox']).float()

        print(f"res: {self.res.shape}")
        print(f"kp2d: {kp2d.shape}")
        print(f"bbox: {bbox.shape}")
        import ipdb; ipdb.set_trace()

        norm_kp2d, _ = self.keypoints_normalizer(
            kp2d[..., :-1].clone(), self.res, self.intrinsics, 224, 224, bbox
        )
        
        # Process image features
        features = torch.cat(self.tracking_results[index]['features'])
        
        # Process initial pose
        init_output = self.smpl.get_output(
            global_orient=self.tracking_results[index]['init_global_orient'],
            body_pose=self.tracking_results[index]['init_body_pose'],
            betas=self.tracking_results[index]['init_betas'],
            pose2rot=False,
            return_full_pose=True
        )
        # print(f"init_output.joints[:, :17]: {init_output.joints[:, :17].shape}")
        # from viz import show_image, COCO_CONNECTIVITY, H36M_CONNECTIVITY, draw_pose_3d_single_frame
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # draw_pose_3d_single_frame(init_output.joints[:, :17], ax=ax, color='b', text=True, azim=0, elev=45, connectivities=COCO_CONNECTIVITY)
        # plt.show()
        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        # plt.close('all')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # draw_pose_3d_single_frame(init_kp3d, ax=ax, color='b', text=True, azim=0, elev=45)
        # plt.show()
        init_kp = torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1)), dim=-1)
        init_smpl = transforms.matrix_to_rotation_6d(init_output.full_pose)
        init_root = transforms.matrix_to_rotation_6d(init_output.global_orient)

        # Process SLAM results
        cam_angvel = convert_dpvo_to_cam_angvel(self.slam_results, self.fps)
        
        to_return = (
            index,                                          # subject id
            self._to(norm_kp2d),                            # 2d keypoints
            (self._to(init_kp), self._to(init_smpl)),       # initial pose
            self._to(features),                             # image features
            self._to(mask),                                 # keypoints mask
            init_root.to(self.device),                      # initial root orientation
            self._to(cam_angvel),                           # camera angular velocity
            self.tracking_results[index]['frame_id'],       # frame indices
            {'cam_intrinsics': self._to(self.intrinsics),   # other keyword arguments
             'bbox': self._to(bbox),
             'res': self._to(self.res)},
            )
        # print("dataset shape")
        print(f"norm_kp2d: {norm_kp2d.shape}")
        # print("2d_keypoints: ", norm_kp2d.shape)
        print("init_kp: ", init_kp.shape)
        # print("init_smpl: ", init_smpl.shape)
        # print("features: ", features.shape)
        # print("mask: ", mask.shape)
        # print()
        import ipdb; ipdb.set_trace()

        return to_return