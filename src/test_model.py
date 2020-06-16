# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : test_model.py
# @Software : PyCharm
import torch
import cv2
import math
from models.MobileFaceNetPruned import MobileFaceNetPv4, MobileFaceNetPv1
from src import transform as trans
import torch.nn.functional as F
import numpy as np
MODEL_MAPPING = {
    'MobileFaceNetPv4':MobileFaceNetPv4,
    'MobileFaceNetPv1':MobileFaceNetPv1
}

class ModelTest():
    def __init__(self, device_id):
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self._define_detector()

    def _define_detector(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def _define_network(self, model_name, patch_info):
        height = int(patch_info.split('x')[0].split('_')[-1])
        width = int(patch_info.split('x')[-1])
        self.kernel_size = ((height+15) // 16, (width+15) // 16)
        self.model = MODEL_MAPPING[model_name](conv6_kernel=self.kernel_size).to(self.device)

    def _load_model(self, model_path, model_name, patch_info):
        self._define_network(model_name, patch_info)
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

    def get_bbox(self, img):
        height, width = img.shape[0],img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img, (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                             interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104,117,123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height,\
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

    def get_prediction(self, img, model_path, model_name, patch_info):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path, model_name, patch_info)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward(img)
            prediction = F.softmax(prediction)
        return prediction.cpu().numpy()








