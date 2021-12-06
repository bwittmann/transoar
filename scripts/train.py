"""Script for training the transoar project."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import monai

from transoar.trainer import Trainer
from transoar.data.dataloader import get_loader
from transoar.utils.io import get_config, write_json, get_meta_data
from transoar.models.transoarnet import TransoarNet
from transoar.models.build import build_criterion


def train(config, args):
    device = config['device']

    # Build necessary components
    train_loader = get_loader(config, 'train')

    if config['overfit']:
        val_loader = get_loader(config, 'train')
    else:
        val_loader = get_loader(config, 'val')

    model = TransoarNet(config).to(device=device)
    criterion = build_criterion(config).to(device=device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": float(config['lr_backbone'])
        },
    ]

    optim = torch.optim.AdamW(
        param_dicts, lr=float(config['lr']), weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['lr_drop'])

    # Load checkpoint if applicable
    if args.resume is not None:
        checkpoint = torch.load(Path(args.resume))

        # Unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        metric_start_val = checkpoint['metric_max_val']
    else:
        epoch = 0
        metric_start_val = 0

    # Init logging
    path_to_run = Path(os.getcwd()) / 'runs' / config['experiment_name']
    path_to_run.mkdir(exist_ok=True)

    # Get meta data and write config to run
    config.update(get_meta_data())
    write_json(config, path_to_run / 'config.json')

    # Build trainer and start training
    trainer = Trainer(
        train_loader, val_loader, model, criterion, optim, scheduler, device, config, 
        path_to_run, epoch, metric_start_val
    )
    trainer.run()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add minimal amount of args (most args should be set in config files)
    parser.add_argument("--config", type=str, required=True, help="Config to use for training located in /config.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default=None)
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)

    # To get reproducable results
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])

    torch.backends.cudnn.benchmark = False  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True

    train(config, args)
