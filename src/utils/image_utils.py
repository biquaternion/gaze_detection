#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Tuple, Union, Iterable

import PIL.Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def get_img_roi(src, inner_points: Iterable[Tuple], logger: logging.Logger = logger) -> Union[Tuple[int, int, int, int], None]:
    threshold, binary = cv2.threshold(src, 4, 255, cv2.THRESH_BINARY)
    it = iter(inner_points)
    inner_point = next(it, None)
    while inner_point is not None:
        x, y = inner_point
        x_vals = np.nonzero(binary[y, :])[0]
        while x_vals.size < 2 and inner_point is not None:
            logger.warning(f'x_vals size = {x_vals.size}, expected > 2')
            inner_point = next(it, None)
            continue
        if inner_point is None:
            return None
        l, r = x_vals[0], x_vals[-1]
        y_vals = np.nonzero(binary[:, x])[0]
        if y_vals.size < 2 and inner_point is not None:
            logger.warning(f'y_vals size = {y_vals.size}, expected > 2')
            inner_point = next(it, None)
            continue

        t, b = y_vals[0], y_vals[-1]
        break
    return l.item(), t.item(), r.item(), b.item()


def add_black_paddings(src: Union[np.ndarray, PIL.Image.Image], target_ratio_wh: Tuple[int, int] = (448, 224)) -> np.ndarray:
    h, w = src.shape[:2]
    tw, th = target_ratio_wh
    if w / h > tw / th:
        new_h = w * th // tw
        new_w = w
    else:
        new_w = h * tw // th
        new_h = h
    result = np.zeros((new_h, new_w, 3), np.uint8)
    result[(new_h - h) // 2:(new_h + h) // 2, (new_w - w) // 2:(new_w + w) // 2] += src
    return result


if __name__ == '__main__':
    images_path = Path('data/GazeT/dataset/images')
    images_paths = [('9492833e-ac73-467d-85c5-5b0dbc09f669', 'step_9.jpeg', [[522, 366], [673, 376], [734, 376],[589, 371]]),
                    ('9492833e-ac73-467d-85c5-5b0dbc09f669', 'step_14.jpeg', [[525, 391], [678, 398], [742, 393], [593, 396]]),
                    ('8f0ba296-96d9-4ff0-8de9-13cc0e2ae671', 'step_1.jpeg', [(543, 698)])] + \
                    [('5db1dd3e-cdb1-4b54-9422-c57e6b60e9dc', f'step_{i}.jpeg', [[1044, 609], [1006, 614], [958, 616], [919, 613]]) for i in range(1, 43)]
    for task, step, inner_points in images_paths:
        image_path = images_path / task / step
        image = cv2.imread(str(image_path))
        roi = get_img_roi(image, inner_points=inner_points, logger=logger)
        print(roi)
        if roi is not None:
            l, t, r, b = roi
            cv2.imshow('image', image[t:b, l:r])
            pad = add_black_paddings(image[t:b, l:r], target_ratio_wh=(2, 1))
            print(l, r, t, b, (r - l) / (b - t), pad.shape[1] / pad.shape[0])
            cv2.imshow('pad', pad)
            cv2.waitKey()
        else:
            logger.warning(f'no roi for {image_path}')
