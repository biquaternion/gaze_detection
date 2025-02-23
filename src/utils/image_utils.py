#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Tuple, Union, Iterable

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


if __name__ == '__main__':
    images_path = Path('data/GazeT/dataset/images')
    images_paths = [('9492833e-ac73-467d-85c5-5b0dbc09f669', 'step_9.jpeg', [[522, 366], [673, 376], [734, 376],[589, 371]]),
                    ('9492833e-ac73-467d-85c5-5b0dbc09f669', 'step_14.jpeg', [[525, 391], [678, 398], [742, 393], [593, 396]]),
                    ('8f0ba296-96d9-4ff0-8de9-13cc0e2ae671', 'step_1.jpeg', [(543, 698)])]
    for task, step, inner_points in images_paths:
        image_path = images_path / task / step
        image = cv2.imread(str(image_path))
        roi = get_img_roi(image, inner_points=inner_points, logger=logger)
        print(roi)
        if roi is not None:
            l, t, r, b = roi
            cv2.imshow('image', image[t:b, l:r])
            cv2.waitKey()
            print(l, r, t, b)
        else:
            logger.warning(f'no roi for {image_path}')
