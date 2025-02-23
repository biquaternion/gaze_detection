#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor

from src.utils.image_utils import get_img_roi

MIN_WIDTH = 60
MAX_WIDTH = 224

DEFAULT_GAZET_PATH = Path('data/GazeT/dataset')

class GazeTDataset(Dataset):
    def __init__(self,
                 data_path: PathLike = DEFAULT_GAZET_PATH,
                 split: str = 'test',
                 transform=None):
        assert split in ['train', 'test'], 'split must be "train" or "test"'
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        self.images_path = self.data_path / 'images'
        self.metadata_path = self.data_path / 'metadata.json'
        self.blacklist_path = self.data_path / 'blacklist.json'
        # self.blacklist = {k: set(v) for k, v in json.load(open(self.blacklist_path, 'r')).items()} if self.blacklist_path.exists() else {}
        self.metadata = {}
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.metadata = list(filter(lambda x: x['split'] == self.split, self.metadata))
        # self.metadata = list(filter(lambda x: x['task_id'] in self.blacklist and x['step'] in self.blacklist[x['task_id']], self.metadata))
        # self.metadata = list(filter(lambda x: x['eyes_left_left'][0] - x['eyes_right_right'][0] > MIN_WIDTH, self.metadata))
        # self.metadata = list(filter(lambda x: x['eyes_left_left'][0] - x['eyes_right_right'][0] < MAX_WIDTH, self.metadata))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        task_id = self.metadata[idx]['task_id']
        step = self.metadata[idx]['step']
        image_path = self.images_path / f'{task_id}' / f'{step}'
        image = Image.open(image_path).convert('RGB')
        relative_x, relative_y = self.metadata[idx]['relative_x'], self.metadata[idx]['relative_y']
        screen_x, screen_y = self.metadata[idx]['screen_size_x'], self.metadata[idx]['screen_size_y']
        eyes_lr = self.metadata[idx]['eyes_left_right']
        eyes_ll = self.metadata[idx]['eyes_left_left']
        eyes_rl = self.metadata[idx]['eyes_right_left']
        eyes_rr = self.metadata[idx]['eyes_right_right']
        center = [(eyes_ll[0] + eyes_rr[0]) // 2, (eyes_ll[1] + eyes_rr[1]) // 2]

        # l, t, r, b = center[0] - 224 // 2, center[1] - 224 // 2, center[0] + 224 // 2, center[1] + 224 // 2
        # l, t, r, b = eyes_rr[0], min(eyes_rr[1], eyes_ll[1]) - (eyes_ll[0] - eyes_rr[0]) // 2, eyes_ll[0], max(eyes_rr[1], eyes_ll[1]) + (eyes_ll[0] - eyes_rr[0]) // 2
        l, t, r, b = get_img_roi(np.asarray(image), [eyes_lr, eyes_rl, eyes_ll, eyes_rr, center])
        cropped_image = image.crop((l + 8, t + 8, r - 8, b - 8))

        targets = [relative_x * screen_x, relative_y * screen_y]
        if self.transform:
            image, targets = self.transform(cropped_image), torch.tensor(targets)
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

