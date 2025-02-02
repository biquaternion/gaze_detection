#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.models as models


class ResnetRegressor(nn.Module):
    def __init__(self, num_outputs: int = 2):
        super(ResnetRegressor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pass