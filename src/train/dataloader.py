#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor

MIN_WIDTH = 60
MAX_WIDTH = 224

DEFAULT_GAZET_PATH = Path('data/GazeT/dataset')
PROCESSED_GAZET_PATH = Path('data/processed/GazeT')

class GazeTDataset(Dataset):
    def __init__(self,
                 data_path: PathLike = PROCESSED_GAZET_PATH,
                 split: str = 'test',
                 transform=None):
        assert split in ['train', 'test'], 'split must be "train" or "test"'
        self.data_path = Path(data_path)
        self.split = split
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
        image = Image.open(image_path).convert('RGB')
        relative_x, relative_y = md['relative_x'], md['relative_y']
        screen_x, screen_y = md['screen_size_x'], md['screen_size_y']

        targets = [relative_x * screen_x, relative_y * screen_y]
        if self.transform:
            image, targets = self.transform(image), torch.tensor(targets).float()
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

