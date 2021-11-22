"""Script for training."""

import numpy as np
import torch
from tqdm import tqdm

from transoar.trainer import Trainer
from transoar.data.dataloader import get_loader
from transoar.utils.io import get_complete_config
from transoar.models.transoarnet import TransoarNet
from transoar.models.build import build_criterion


def train(config):
    device = config['training']['device']

    # Build necessary components
    train_loader = get_loader(config['data'], 'train')
    val_loader = get_loader(config['data'], 'train')

    model = TransoarNet(config['model'], config['data']['num_classes']).to(device=device)
    criterion = build_criterion(config['training']).to(device=device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": float(config['training']['lr_backbone'])
        },
    ]

    optim = torch.optim.AdamW(
        param_dicts, lr=float(config['training']['lr']), weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['training']['lr_drop'])

    # Build trainer and start training
    trainer = Trainer(
        train_loader, val_loader, model, criterion, optim, scheduler, device, config
    )
    trainer.run()
        

if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

    config = get_complete_config()
    train(config)
      