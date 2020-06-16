"""
原始图片生成训练patch
根据人脸bbox坐标，计算仿射变换矩阵的参数，生成相应的patch
"""
import cv2
import numpy as np


class AffineCrop():

    def __init__(self, bbox, width, height, scale=1., shift_ratio_x=0., shift_ratio_y=0.):

        self.bbox = bbox
        self.width = width
        self.height = height
        self.scale = scale
        self.shift_ratio_x = shift_ratio_x
        self.shift_ratio_y = shift_ratio_y

    def _get_center(self, x, y, width, height):

        return width / 2 + x, height / 2 + y

    def _get_new_box(self, src_w, src_h):

        x = self.bbox[0]
        y = self.bbox[1]
        box_w = self.bbox[2]
        box_h = self.bbox[3]

        shift_x = box_w * self.shift_ratio_x
        shift_y = box_h * self.shift_ratio_y

        self.scale = min((src_h - 1)/box_h, min((src_w - 1)/box_w, self.scale))

        new_width = box_w * self.scale
        new_height = box_h * self.scale

        center_x, center_y = self._get_center(x, y, box_w, box_h)

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

    def crop(self, org_img, crop=True):
        if not crop:
            dst_img = cv2.resize(org_img, (self.width, self.height))
        else:
            h, w, _ = np.shape(org_img)

            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(w, h)

            img = org_img[left_top_y: right_bottom_y, left_top_x: right_bottom_x]

            dst_img = cv2.resize(img, (self.width, self.height))

        return dst_img








