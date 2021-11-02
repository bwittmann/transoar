"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset

def get_train_loader(data_config):
    data_set = TransoarDataset(data_config, 'train')

def get_val_loader(data_config):
    data_set = TransoarDataset(data_config, 'val')

def get_test_loader(data_config):
    data_set = TransoarDataset(data_config, 'test')