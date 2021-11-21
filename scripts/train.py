"""Script for training."""

import numpy as np
import torch
from tqdm import tqdm

from transoar.data.dataloader import get_loader
from transoar.utils.io import get_complete_config
from transoar.models.build import build_model


def train(config):
    loader = get_loader(config['data'], 'train')
    device = config['training']['device']
    model = build_model(config['model']).to(device=device)
    
    for data, mask, bboxes, seg_labels in tqdm(loader):
        data, mask = data.to(device=device), mask.to(device=device)
        out = model(data, mask)

        k = 12
        

if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

    config = get_complete_config()

    train(config)
      