#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from src.train.dataloader import GazeTDataset
from src.train.regressor import ResnetRegressor

BATCH_SIZE = 32
PROJECT_NAME = 'GazeT-Regressor'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainRegressor:
    def __init__(self):
        self.run = wandb.init(project=PROJECT_NAME, name=__name__)
        self.logger = logging.getLogger(wandb.__name__)

    def train(self, model: torch.nn.Module,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              criterion: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              scheduler,
              num_epochs,
              device):
        self.logger.info('start training')
        for epoch in range(num_epochs):
            self.logger.info(f'epoch {epoch + 1}/{num_epochs}')
            model.train()
            train_loss = 0.0
            self.logger.info('train loop')
            for i, (inputs, targets) in enumerate(tqdm(train_dataloader)):
                train_loss += self.train_step(criterion, device, inputs, model, optimizer, targets)
            train_loss /= len(train_dataloader)
            wandb.log({'train_loss': train_loss})
            model.eval()
            val_loss = 0
            with torch.no_grad():
                self.logger.info('validation')
                for i, (inputs, targets) in enumerate(tqdm(val_dataloader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    val_loss += self.val_step(model(inputs), targets, criterion)
                val_loss /= len(val_dataloader)
            wandb.log({'val_loss': val_loss})
            scheduler.step()
        torch.save(model.state_dict(), 'model.pth')
        wandb.save('model.pth')
        wandb.finish()

    def train_step(self, criterion, device, inputs, model, optimizer, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss

    def val_step(self, predictions, targets, criterion):
        return criterion(predictions, targets).item()


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: DictConfig):
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    learning_rate = cfg.train.learning_rate
    logger.info(f'config:\n{cfg}')
    cwd = hydra.utils.get_original_cwd()
    print(cwd)
    cwd = Path(cwd)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    logger.info('instantiating model')
    model = ResnetRegressor(num_outputs=2).to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    logger.info(f'preparing train dataset')
    train_dataset = GazeTDataset(data_path=cwd / 'data' / 'processed' / 'GazeT',
                                 split='train',
                                 transform=transform,
                                 fit_size_policy=cfg.dataset.fit_size_policy)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info(f'preparing val dataset')
    val_dataset = GazeTDataset(data_path=cwd / 'data' / 'processed' / 'GazeT',
                               split='test',
                               transform=transform,
                               fit_size_policy=cfg.dataset.fit_size_policy)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    logger.info(f'start training')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_regressor = TrainRegressor()
    train_regressor.train(model, train_dataloader, val_dataloader,
                          criterion, optimizer, scheduler,
                          num_epochs=num_epochs, device=device)


if __name__ == '__main__':
    main()
