import os, sys
import yaml
import torch
from loguru import logger

from configs import constants as _C
from .smpl import SMPL


def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model


def build_motion_encoder(cfg):
    from lib.models.layers import MotionEncoder

    with open(cfg.MODEL_CONFIG, 'r') as f:
        model_config = yaml.safe_load(f)
    model_config.update({'d_feat': _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]})

    d_embed = model_config['d_embed']
    pose_dr = model_config['pose_dr']
    rnn_type = model_config['layer']
    n_layers = model_config['n_layers']
    n_joints = _C.KEYPOINTS.NUM_JOINTS
    in_dim = n_joints * 2 + 3

    # Module 1. Motion Encoder
    network = MotionEncoder(in_dim=in_dim,
                            d_embed=d_embed,
                            pose_dr=pose_dr,
                            rnn_type=rnn_type,
                            n_layers=n_layers,
                            n_joints=n_joints).to(cfg.DEVICE)

    # Load Checkpoint
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        model_state_dict = {k.split('motion_encoder.')[-1]: v
                            for k, v in checkpoint['model'].items() if 'motion_encoder' in k}
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"=> loaded checkpoint '{cfg.TRAIN.CHECKPOINT}' ")
    else:
        logger.info(f"=> Warning! no checkpoint found at '{cfg.TRAIN.CHECKPOINT}'.")

    return network


def build_network(cfg, smpl):
    from .wham import Network
    
    with open(cfg.MODEL_CONFIG, 'r') as f:
        model_config = yaml.safe_load(f)
    model_config.update({'d_feat': _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]})
    
    network = Network(smpl, **model_config).to(cfg.DEVICE)
    
    # Load Checkpoint
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"=> loaded checkpoint '{cfg.TRAIN.CHECKPOINT}' ")
    else:
        logger.info(f"=> Warning! no checkpoint found at '{cfg.TRAIN.CHECKPOINT}'.")
        
    return network