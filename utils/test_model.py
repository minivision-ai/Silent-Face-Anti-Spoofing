# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : test_model.py
# @Software : PyCharm
import torch
import os
import numpy as np
from models.MobileFaceNet_pruned import MobileFaceNetPv4, MobileFaceNetPv1
from utils import transform as trans
import torch.nn.functional as F
MODEL_MAPPING = {
    'MobileFaceNetPv4':MobileFaceNetPv4,
    'MobileFaceNetPv1':MobileFaceNetPv1
}

class ModelTest():
    def __init__(self, device_id):
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    def define_network(self, model_name, patch_info):
        height = int(patch_info.split('x')[0].split('_')[-1])
        width = int(patch_info.split('x')[-1])
        self.kernel_size = ((height+15) // 16, (width+15) // 16)
        self.model = MODEL_MAPPING[model_name](conv6_kernel=self.kernel_size).to(self.device)

    def load_model(self, model_path, model_name, patch_info):
        self.define_network(model_name, patch_info)
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def get_prediction(self, img, model_path, model_name, patch_info):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.load_model(model_path, model_name, patch_info)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward(img)
            prediction = F.softmax(prediction)
        return prediction.cpu().numpy()








