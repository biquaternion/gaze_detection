#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest

from src.utils.image_utils import get_img_roi, add_black_paddings


@pytest.fixture(scope="module")
def shape():
    return 640, 480, 3


@pytest.fixture(scope="module")
def roi_coords(shape):
    w, h = 100, 50
    t, l = np.random.randint(shape[0] - h), np.random.randint(shape[1] - w)
    return l, t, w, h


@pytest.fixture(scope="module")
def roi_random_coords(shape):
    w, h = random.randint(50, 100), random.randint(50, 100)
    t, l = np.random.randint(shape[0] - h), np.random.randint(shape[1] - w)
    return l, t, w, h


@pytest.fixture(scope="module")
def image(shape, roi_coords):
    canvas = np.zeros(shape=shape, dtype=np.uint8)
    l, t, w, h = roi_coords
    canvas[t:t + h, l:l + w] = np.random.randint(1, 255, size=(h, w, 3), dtype=np.uint8)
    return canvas


def test_get_img_roi(image, roi_coords):
    expected = roi_coords
    inner_point = roi_coords[0] + 50, roi_coords[1] + 25
    l, t, r, b = get_img_roi(image, inner_point)
    el, et, ew, eh = expected
    assert (l, t, r - l + 1, b - t + 1) == (el, et, ew, eh)


def test_add_black_paddings(image, roi_coords):
    l, t, w, h = roi_coords
    new_image = add_black_paddings(src=image, target_ratio_wh=(1, 1))
    tolerance = 1e-3
    nh, nw, nc = new_image.shape
    assert abs(nh / nw - 1) < tolerance


if __name__ == '__main__':
    pass
