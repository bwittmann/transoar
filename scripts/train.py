"""Script for training."""

import numpy as np
import torch
from tqdm import tqdm

from transoar.data.dataloader import get_loader
from transoar.utils.io import get_complete_config
from transoar.models.main_net import TransoarNet


def train(config):
    loader = get_loader(config['data'], 'train')
    model = TransoarNet(config['model']['swin_tiny'])

    for data, mask, bboxes, seg_labels in tqdm(loader):
        out = model(data)
        

if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

    config = get_complete_config()

    train(config)
      