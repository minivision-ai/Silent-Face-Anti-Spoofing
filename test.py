# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
from src.test_model import ModelTest
from src.generate_patches import AffineCrop
import os
import cv2
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def test(image_name, model_dir, device_id):
    model_test = ModelTest(device_id)
    image = cv2.imread('./images/'+image_name)
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    # get the prediction of every model in the combined models
    for model in os.listdir(model_dir):
        info = model.split('_')[0:-1]
        patch_info = '_'.join(info)
        height, width = info[-1].split('x')
        model_name = model.split('.pth')[0].split('_')[-1]
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
    # save prediction result
    label = np.argmax(prediction)
    if label == 1:
        print("True Face")
        result_text = "True"
        color = (255, 0, 0)
    else:
        print("False Face")
        result_text = "False"
        color = (0, 0, 255)
    cv2.rectangle(image, (image_bbox[0], image_bbox[1]),
                  (image_bbox[0]+image_bbox[2], image_bbox[1]+image_bbox[3]), color,2)
    cv2.putText(image, result_text, (image_bbox[0], image_bbox[1]-5),cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
    cv2.imwrite("./result/"+image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./resources/combined_models", help="models used to test")
    parser.add_argument("--image_name", type=str, default="image_T1.jpg", help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)

