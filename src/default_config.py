# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training and test

"""
import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist

def get_default_config():

    conf = EasyDict()

    # ----------------------training---------------
    conf.schedule_lr_type = "MSTEP"#warmup
    conf.lr = 1e-1
    # [9, 13, 15]
    conf.milestones = [24, 40, 48]  # down learing rate
    conf.gamma = 0.1
    conf.epochs =60
    conf.momentum = 0.9
    conf.batch_size = 1024

    # model
    conf.net_mode = 'MultiFTNet'
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128
    # FT
    conf.ft_height = 10
    conf.ft_width = 10
    conf.ft_root = '/ssd/data/recognize_data/LiveBody/Train/new_patches/Fourier'

    # dataset
    conf.train_root_path = '/ssd/data/recognize_data/LiveBody/Train/new_patches'

    # save file path
    conf.snapshot_root = '/gpfs10/user_home/zhuying/Models'
    conf.OS_snapshot_dir_path = '{}/LiveBody/snapshot'.format(conf.snapshot_root)

    # log path
    conf.log_path = '/ssd/user_home/zhuying/LiveBody/jobs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 30

    return conf


def update_config(args, conf, file_name):

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    w_input = int(args.patch_info.split('x')[-1])
    h_input = int(args.patch_info.split('x')[0].split('_')[-1])
    conf.input_size = [h_input, w_input]
    conf.kernel_size = ((h_input+15)//16, (w_input+15)//16)
    conf.num_classes = args.num_classes
    conf.patch_info = args.patch_info
    conf.train_set = args.train_set

    conf.devices = args.devices
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # set ArcFace loss param  bias, scale_value
    job_name = file_name.split('-c')[0]
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)

    snapshot_dir = '{}/{}'.format(conf.OS_snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    # save directory path
    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf