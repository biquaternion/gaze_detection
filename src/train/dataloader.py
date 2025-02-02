#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from os import PathLike
from pathlib import Path

import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor

DEFAULT_GAZET_PATH = Path('data/GazeT/dataset')

class GazeTDataset(Dataset):
    def __init__(self,
                 data_path: PathLike = DEFAULT_GAZET_PATH,
                 transform=None):
        self.data_path = Path(data_path)
        self.transform = transform

        self.images_path = self.data_path / 'images'
        self.metadata_path = self.data_path / 'metadata.json'
        self.metadata = {}
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        task_id = self.metadata[idx]['task_id']
        step = self.metadata[idx]['step']
        image_path = self.images_path / f'{task_id}' / f'{step}'
        image = Image.open(image_path).convert('RGB')
        relative_x, relative_y = self.metadata[idx]['relative_x'], self.metadata[idx]['relative_y']

        # image, targets = pil_to_tensor(image), torch.tensor([relative_x, relative_y])
        targets = [relative_x, relative_y]
        if self.transform:
            image, targets = self.transform(image), torch.tensor(targets)
        return image, targets



if __name__ == '__main__':
    images_path = DEFAULT_GAZET_PATH / 'images'
    metadata_path = DEFAULT_GAZET_PATH / 'metadata.json'
    metadata = json.load(open(metadata_path, 'r'))
