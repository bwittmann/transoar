"""Script for training."""

import numpy as np
import torch
from tqdm import tqdm

from transoar.data.dataloader import get_loader
from transoar.utils.io import get_complete_config
from transoar.models.transoarnet import TransoarNet


def train(config):
    loader = get_loader(config['data'], 'train')
    model = TransoarNet(config['model']).to(device=config['training']['device'])
    
    for data, mask, bboxes, seg_labels in tqdm(loader):
        data = data.to(device=config['training']['device'])

        out = model(data)
        

if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

    config = get_complete_config()

    train(config)
      