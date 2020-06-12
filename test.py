# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
from utils.test_model import ModelTest
from utils.generate_patches import AffineCrop
import os
import cv2
import numpy as np
import argparse

IMAGE_BBOX={
    'image_F1.jpg':[212, 327, 297, 339],
    'image_T1.jpg':[223, 379, 264, 378]
}


def test(image_name, model_dir, device_id):
    model_test = ModelTest(device_id)
    image_bbox = IMAGE_BBOX[image_name]
    image = cv2.imread('./images/'+image_name)
    prediction = np.zeros((1, 3))
    for model in os.listdir(model_dir):
        info = model.split('_')[0:-1]
        patch_info = '_'.join(info)
        height, width = info[-1].split('x')
        model_name = model.split('.pth')[0].split('_')[-1]
        print(patch_info, model_name)
        if patch_info.find('org') >= 0:
            crop = False
            image_crop = AffineCrop(image_bbox, int(width), int(height))
            img = image_crop.crop(image, crop)
        else:
            info = patch_info.split('_')
            scale, shift_x, shift_y = float(info[0]), float(info[1]), float(info[2])
            image_crop = AffineCrop(image_bbox, int(width), int(height), scale, shift_x, shift_y)
            img = image_crop.crop(image)
        prediction += model_test.get_prediction(img, os.path.join(model_dir, model), model_name, patch_info)
    print(prediction)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=6, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./combine_models", help="models used to test")
    parser.add_argument("--image_name", type=str, default="image_T1.jpg", help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)

