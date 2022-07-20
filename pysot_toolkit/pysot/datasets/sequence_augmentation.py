# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from pysot.utils.bbox import corner2center, \
        Center, center2corner, Corner


class Augmentation:
    def __init__(self, shift, scale, blur, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz[0]-1) / (bbox[2]-bbox[0])
        b = (out_sz[1]-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, out_sz,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _shift_scale_aug(self, image, bbox, mask, crop_bbox):
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            bbox_center = corner2center(bbox)
            scale_x = max(scale_x, bbox_center.w / crop_bbox_center.w)
            scale_y = max(scale_y, bbox_center.h / crop_bbox_center.h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            bbox_center = corner2center(bbox)
            max_shift = min((crop_bbox_center.w+crop_bbox_center.h)/2*self.shift,
                            bbox_center.w + bbox_center.h)
            sx = Augmentation.random() * max_shift
            sy = Augmentation.random() * max_shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(bbox.x2 - x2, min(bbox.x1 - x1, sx))
            sy = max(bbox.y2 - y2, min(bbox.y1 - y1, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        avg_chans = np.mean(image, axis=(0, 1))
        image = self._crop_roi(image, crop_bbox, (im_w, im_h), padding=avg_chans)
        if mask is not None:
            mask = self._crop_roi(mask, crop_bbox, (im_w, im_h))
        return image, bbox, mask

    def __call__(self, image, bbox, mask=None):
        shape = image.shape

        crop_bbox = Corner(0, 0, shape[1], shape[0])
        bbox = Corner(bbox[0], bbox[1], bbox[2], bbox[3])

        # shift scale augmentation
        image, bbox, mask = self._shift_scale_aug(image, bbox, mask, crop_bbox)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # print(image.shape, bbox)

        return image, np.array(bbox), mask
