# -*- coding: utf-8 -*-
# @Time : 20-6-11 上午10:16
# @Author : zhuying
# @Company : Minivision
# @File : count_speed_flops.py
# @Software : PyCharm
import time
import torch
import numpy as np
from tools.iccv_flops_calculation import profile
from models.MobileFaceNet_pruned import MobileFaceNetPv4, MobileFaceNetPv1, MobileFaceNetPv2, MobileFaceNetPv3,MobileFaceNet
MODEL_MAPPING = {
    'MobileFaceNet':MobileFaceNet,
    'MobileFaceNetPv4':MobileFaceNetPv4,
    'MobileFaceNetPv1':MobileFaceNetPv1,
    'MobileFaceNetPv2':MobileFaceNetPv2,
    'MobileFaceNetPv3':MobileFaceNetPv3
}


class CalculateFlopSpeed():
    def __init__(self, model_info, input_size, device_id):
        self.input = torch.randn(input_size)
        self.input_size = input_size
        self.model = MODEL_MAPPING[model_info](embedding_size=128, conv6_kernel=(5,5))
        self.device = "cuda:{}".format(device_id if torch.cuda.is_available() else "cpu")
        if device_id>0:
            self.cuda = True
            self.model = self.model.to(self.device)
            self.input = self.input.to(self.device)

    def compute_time(self):
        self.model.eval()
        i = 0
        time_spent = []
        while i < 400:
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(self.input)
            if self.cuda:
                torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            if i != 0:
                time_spent.append(time.time() - start_time)
            i += 1
        print('Forward time per img (batch_size=%d): %.2fms, (Mean: %.5fms)' %
              (self.input.size()[0], np.mean(time_spent) * 1e3,
               sum(time_spent) / len(time_spent) / self.input.size()[0] * 1e3))

    def compute_flop_iccv(self):
        flops, params = profile(self.model, input_size=self.input_size)
        print(
            'flops:{:.5f}G , params:{:.3f}M'.format(
                (flops / 1e9),
                params / 1e6))


if __name__ == "__main__":
    param = {
        "model_info": "MobileFaceNet",
        "input_size": (1, 3, 80, 80),
        "device_id":  6,
    }
    cal = CalculateFlopSpeed(**param)
    cal.compute_flop_iccv()
    cal.compute_time()