# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
from torchvision import datasets


def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, ft_root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_root = ft_root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # get the FT picture of the sample
        ft_path = path.replace(self.root, self.ft_root)
        ft_sample = self.loader(ft_path)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.cvtColor(ft_sample, cv2.COLOR_BGR2GRAY)
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target
