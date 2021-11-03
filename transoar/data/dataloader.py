"""Module containing dataloader related functionality."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.io import get_config
from transoar.utils.bboxes import segmentation2bbox

data_config = get_config('data')

def get_loader(data_config, split):
    dataset = TransoarDataset(data_config, split)
    dataloader = DataLoader(
        dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'], collate_fn=pad_collate
    )
    return dataloader

def pad_collate(batch):
    # Estimate max shape for padding
    shapes = []
    for image, _ in batch:
        shapes.append(image.shape)

    max_shape = torch.max(torch.tensor(shapes), axis=0)[0]
    
    # Pad to form a batch with equal shapes
    batch_images = []
    batch_labels = []
    batch_masks = []
    for image, label in batch:
        to_pad = max_shape - torch.tensor(image.shape)
        padding = [0, to_pad[-1], 0, to_pad[-2], 0, to_pad[-3]]
        batch_images.append(F.pad(image, padding, 'constant', 0))
        batch_labels.append(F.pad(label, padding, 'constant', 0))
        batch_masks.append(F.pad(torch.zeros_like(image), padding, 'constant', -1))

    # Generate bboxes and corresponding class labels
    batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), data_config['bbox_padding'])

    return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)