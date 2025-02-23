#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd
import cv2
from tqdm import tqdm

from src.utils.image_utils import get_img_roi


def convert_dataset():
    data_path = Path('data')
    dataset_path = data_path / 'GazeT' / 'dataset'
    metadata_path = dataset_path / 'metadata.json'
    result = []
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        processed_path = data_path / 'processed' / 'GazeT'
        processed_path.mkdir(parents=True, exist_ok=True)
        (processed_path / 'images').mkdir(parents=True, exist_ok=True)
        for md in tqdm(metadata):
            eyes_lr = md['eyes_left_right']
            eyes_ll = md['eyes_left_left']
            eyes_rl = md['eyes_right_left']
            eyes_rr = md['eyes_right_right']
            center = [(eyes_ll[0] + eyes_rr[0]) // 2, (eyes_ll[1] + eyes_rr[1]) // 2]
            img = cv2.imread(dataset_path / 'images' / md['task_id'] / md['step'])
            roi = get_img_roi(img, [center, eyes_lr, eyes_rl, eyes_rr, eyes_ll])
            if roi is None:
                continue
            l, t, r, b = roi
            res_img = img[t:b, l:r]
            res_img_path = processed_path / 'images' / md['task_id']
            res_img_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(res_img_path / md['step'], res_img)
            md_ = {k: v for k, v in md.items()}
            md_['roi'] = [l, t, r, b]
            result.append(md_)
    if result:
        pd.DataFrame(result).to_csv(processed_path / 'metadata.csv')


if __name__ == '__main__':
    convert_dataset()
