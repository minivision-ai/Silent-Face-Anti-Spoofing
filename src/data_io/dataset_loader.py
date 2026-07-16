# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io.transform import ToTensor


def get_train_loader(conf):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=tuple(conf.input_size),
                                     scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        ToTensor()
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return train_loader
