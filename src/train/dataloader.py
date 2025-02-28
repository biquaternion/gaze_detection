#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from os import PathLike
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.utils.image_utils import add_black_paddings

MIN_WIDTH = 60
MAX_WIDTH = 224

DEFAULT_GAZET_PATH = Path('data/GazeT/dataset')
PROCESSED_GAZET_PATH = Path('data/processed/GazeT')

class GazeTDataset(Dataset):
    def __init__(self,
                 data_path: PathLike = PROCESSED_GAZET_PATH,
                 split: str = 'test',
                 fit_size_policy: str = 'resize',
                 target_size: Tuple[int, int] = (448, 224),
                 transform=None):
        assert split in ['train', 'test'], 'split must be "train" or "test"'
        self.data_path = Path(data_path)
        self.split = split
        self.fit_size_policy = fit_size_policy
        self.target_size = target_size
        self.transform = transform

        self.images_path = self.data_path / 'images'
        self.metadata_path = self.data_path / 'metadata.csv'
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        md = self.metadata.iloc[idx]
        task_id = md['task_id']
        step = md['step']
        image_path = self.images_path / f'{task_id}' / f'{step}'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        relative_x, relative_y = md['relative_x'], md['relative_y']
        screen_x, screen_y = md['screen_size_x'], md['screen_size_y']

        if self.fit_size_policy == 'resize':
            image = cv2.resize(image, dsize=self.target_size)
        elif self.fit_size_policy == 'padding':
            image = add_black_paddings(src=image, target_ratio_wh=(self.target_size[0], self.target_size[1]))
        targets = [relative_x, relative_y]
        if self.transform:
            image, targets = self.transform(Image.fromarray(image)), torch.tensor(targets).float()
        return image, targets


if __name__ == '__main__':
    images_path = DEFAULT_GAZET_PATH / 'images'
    metadata_path = DEFAULT_GAZET_PATH / 'metadata.json'
    metadata = json.load(open(metadata_path, 'r'))

    transform = transforms.Compose([transforms.Resize((224, 448)), transforms.ToTensor()])
    dataset = GazeTDataset(split='train', transform=transform)
    print(len(dataset))
    for idx, (image, targets) in enumerate(dataset):
        image = np.array(image).transpose((1, 2, 0))
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

