"""Module containing dataloader related functionality."""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transoar.data.dataset import TransoarDataset
from transoar.utils.io import get_config, load_json
from transoar.utils.bboxes import segmentation2bbox


try:
    data_config = get_config('data_main')
    path_to_max_shape = Path(os.getcwd()) / 'dataset' / (data_config['dataset_name'] + '_' + data_config['modality']) / 'max_size.json'
    max_shape = load_json(path_to_max_shape)['max_size']
except:
    max_shape = None

def get_loader(data_config, split):
    dataset = TransoarDataset(data_config, split)
    dataloader = DataLoader(
        dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'], collate_fn=pad_collate
    )
    return dataloader

def pad_collate(batch):
    max_shape_for_padding = max_shape

    if max_shape_for_padding is None:
    # Estimate max shape in the batch for padding
        shapes = []
        for image, _ in batch:
            shapes.append(image.shape)

        max_shape_for_padding = torch.max(torch.tensor(shapes), axis=0)[0]
    else:
         max_shape_for_padding = torch.tensor(max_shape)
    
    # Pad to form a batch with equal shapes
    batch_images = []
    batch_labels = []
    batch_masks = []
    for image, label in batch:
        to_pad = max_shape_for_padding - torch.tensor(image.shape)
        padding = [0, to_pad[-1], 0, to_pad[-2], 0, to_pad[-3]]
        batch_images.append(F.pad(image, padding, 'constant', 0))
        batch_labels.append(F.pad(label, padding, 'constant', 0))
        batch_masks.append(F.pad(torch.zeros_like(image), padding, 'constant', -1))

    # Generate bboxes and corresponding class labels
    batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), data_config['bbox_padding'])

    return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)