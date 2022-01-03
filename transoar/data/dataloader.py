"""Module containing dataloader related functionality."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.bboxes import segmentation2bbox


def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = TransoarCollator(config)

    dataset = TransoarDataset(config, split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=config['shuffle'],
        num_workers=config['num_workers'], collate_fn=collator
    )
    return dataloader

# def init_fn(worker_id):
#     """https://github.com/pytorch/pytorch/issues/7068"""
#     np.random.seed(10 + worker_id)


class TransoarCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']

    def __call__(self, batch):
        batch_images = []
        batch_labels = []
        batch_masks = []
        for image, label in batch:
            batch_images.append(image)
            batch_labels.append(label)
            batch_masks.append(torch.zeros_like(image))

        # Generate bboxes and corresponding class labels
        batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)

        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)
