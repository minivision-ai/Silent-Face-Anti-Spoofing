# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm
from torch.utils.data import DataLoader
from src.dataset_folder import opencv_loader, DatasetFolderFT
from src import transform as trans
from torchvision import datasets


def get_test_loader(conf):
    test_transform = trans.Compose([
                trans.ToTensor(),
            ])
    root_path = '{}/{}'.format(conf.test_root_path, conf.patch_info)
    testset = datasets.ImageFolder(
        root_path,
        test_transform,
        None,
        opencv_loader
        )
    test_loader = DataLoader(
        testset,
        batch_size=conf.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )
    return test_loader


def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path, conf.ft_root, train_transform, None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return train_loader