#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def get_img_roi(src, inner_point: Tuple, refine: bool = True):
    x, y = inner_point
    x_vals = np.nonzero(src[y, :])[0]
    l, r = x_vals[0], x_vals[-1]
    y_vals = np.nonzero(src[:, x])[0]
    t, b = y_vals[0], y_vals[-1]
    if refine:
        threshold, binary = cv2.threshold(src[t:b, l:r, 0], thresh=10, maxval=255, type=cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
        x -= l
        y -= t
        x_vals = np.nonzero(binary[y, :])[0]
        l += x_vals[0]
        r = l + x_vals[-1] - x_vals[0]
        y_vals = np.nonzero(binary[:, x])[0]
        t += y_vals[0]
        b = t + y_vals[-1] - y_vals[0]
    return l.item(), t.item(), r.item(), b.item()


if __name__ == '__main__':
    images_path = Path('data/GazeT/dataset/images')
    image_path = images_path / '8f0ba296-96d9-4ff0-8de9-13cc0e2ae671' / 'step_1.jpeg'
    image = cv2.imread(str(image_path))
    l, r, t, b = get_img_roi(image, inner_point=(543, 698))
    cv2.imshow('image', image[t:b, l:r])
    cv2.waitKey()
    print(l, r, t, b)
