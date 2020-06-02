
"""
原始图片生成训练patch
根据人脸bbox坐标，计算仿射变换矩阵的参数，生成相应的patch
"""
import os
import cv2
import numpy as np


class Affine_Crop():

    def __init__(self, bbox, width, height, scale=1., shift_ratio_x=0., shift_ratio_y=0.):

        self.bbox = bbox
        self.width = width
        self.height = height
        self.scale = scale
        self.shift_ratio_x = shift_ratio_x
        self.shift_ratio_y = shift_ratio_y

    def get_center(self, x, y, width, height):

        return width / 2 + x, height / 2 + y

    def get_new_box(self, src_w, src_h):

        x = self.bbox[0]
        y = self.bbox[1]
        box_w = self.bbox[2]
        box_h = self.bbox[3]

        shift_x = box_w * self.shift_ratio_x
        shift_y = box_h * self.shift_ratio_y

        self.scale = min((src_h - 1)/box_h, min((src_w - 1)/box_w, self.scale))

        new_width = box_w * self.scale
        new_height = box_h * self.scale

        center_x, center_y = self.get_center(x, y, box_w, box_h)

        left_top_x = center_x - new_width / 2 + shift_x
        left_top_y = center_y - new_height / 2 + shift_y
        right_bottom_x = center_x + new_width / 2 + shift_x
        right_bottom_y = center_y + new_height / 2 + shift_y

        if left_top_x < 0:

            s = left_top_x
            left_top_x -= s
            right_bottom_x -= s

        if left_top_y < 0:

            s = left_top_y
            left_top_y -= s
            right_bottom_y -= s

        if right_bottom_x > src_w:

            s = right_bottom_x -src_w
            left_top_x -= s
            right_bottom_x -= s

        if right_bottom_y > src_h:
            s = right_bottom_y - src_h
            left_top_y -= s
            right_bottom_y -= s

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img):

        h, w, _ = np.shape(org_img)

        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self.get_new_box(w, h)

        img = org_img[left_top_y: right_bottom_y, left_top_x: right_bottom_x]

        dst_img = cv2.resize(img, (self.width, self.height))

        return dst_img


if "__main__" == __name__:

    root_path = ''
    save_root_path = ''
    landmark_file_path = ''
    f = open(landmark_file_path, 'r')
    lines = f.readlines()

    patch_list = ['1_0_0_80x80', '2.7_0_0_80x80', '4_0_0_80x80', '2.7_0.1_0_80x80', '2.7_0.2_0.1_80x80']

    for patch in patch_list:

        scale_, shift_x, shift_y, area = patch.split('_')

        height_, width_ = area.split('x')

        save_path = os.path.join(save_root_path, patch)

        for line in lines:

            list_ = line.split(' ')

            if len(list_) != 16:
                print(line)
                continue

            img = cv2.imread(os.path.join(root_path, list_[0]))

            bbox = [int(list_[1]), int(list_[2]), int(list_[3]), int(list_[4])]

            g = Affine_Crop(bbox, int(width_), int(height_), float(scale_), float(shift_x), float(shift_y))
            img = g.crop(img)
            img_save_path = os.path.join(save_path, list_[0])
            parent_path = os.path.dirname(img_save_path)
            if not os.path.isdir(parent_path):
                print(parent_path)
                os.makedirs(parent_path)
            cv2.imwrite(img_save_path, img)







