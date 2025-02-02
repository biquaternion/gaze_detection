#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.train.dataloader import GazeTDataset
from src.train.regressor import ResnetRegressor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_regressor(model: torch.nn.Module,
                    dataloader,
                    criterion,
                    optimizer,
                    scheduler,
                    num_epochs,
                    device):
    model.train()
    logger.info('start training')
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            logging.warning(f'batch: {i}/{len(dataloader)}, loss: {loss.item():.4f}')
        scheduler.step()
        logging.warning(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss.item()}')




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResnetRegressor(num_outputs=2).to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = GazeTDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_regressor(model, dataloader, criterion, optimizer, scheduler, num_epochs=2, device=device)