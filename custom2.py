from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import scipy.stats as stats

from configs import constants as _C
from lib.data.normalizer import Normalizer
from lib.utils import transforms
from lib.models import build_body_model
from lib.utils.kp_utils import root_centering
from lib.utils.imutils import compute_cam_intrinsics

KEYPOINTS_THR = 0.3


def xyxy_to_cxcys(bbox):
    cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    assert scale > 0, f"scale: {scale}"
    return np.array([[cx, cy, scale]])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, kp_data, tracking_results, smpl):
        self.device = cfg.DEVICE.lower()

        self.smpl = build_body_model('cpu')
        self.kp_data = kp_data
        self.keypoints_normalizer = Normalizer(cfg)
        self.tracking_results = tracking_results
        # self.smpl = smpl

        self._to = lambda x: x.unsqueeze(0).to(self.device)

    def __len__(self):
        return len(self.tracking_results.keys())

    def __getitem__(self, _index):
        if _index >= len(self): return

        # Process 2D keypoints
        ped_id, frames = list(self.kp_data.items())[_index]
        frames = [x[1] for x in sorted(frames.items(), key=lambda x: x[0])]
        height = stats.mode([a['height'] for a in frames])[0]
        width = stats.mode([a['weight'] for a in frames])[0]
        res = torch.tensor([width, height]).float()
        kp2d = torch.tensor(np.array([np.hstack([a['pose'], a['score'][:, None]]) for a in frames])).float()
        bboxes = [a.get('box', None) for a in frames]
        if np.all(bboxes is not None):
            bbox = np.concatenate([xyxy_to_cxcys(b) for b in bboxes])
            bbox = torch.tensor(bbox).float()
        else:
            bbox = None

        norm_kp2d, _ = self.keypoints_normalizer(
                kp2d[..., :-1].clone(), res, None, 224, 224, bbox
        )

        # Process initial pose
        cam_id = frames[0]['cam_id']
        init_output = self.smpl.get_output(
                global_orient=self.tracking_results[cam_id][ped_id]['init_global_orient'],
                body_pose=self.tracking_results[cam_id][ped_id]['init_body_pose'],
                betas=self.tracking_results[cam_id][ped_id]['init_betas'],
                pose2rot=False,
                return_full_pose=True
        )
        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        init_kp = torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1)), dim=-1)

        print(f"norm_kp2d: {norm_kp2d.shape}")
        print(f"init_kp: {init_kp.shape}")
        import ipdb; ipdb.set_trace()
        to_return = (
                ped_id,  # subject id
                self._to(norm_kp2d),  # 2d keypoints
                self._to(init_kp)  # initial pose
        )
        return to_return