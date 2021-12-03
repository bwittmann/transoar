"""Module containing dataloader related functionality."""

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


class TransoarCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']
        self._max_shape = config['max_size'] if 'max_size' in config else None

    def __call__(self, batch):
        if self._max_shape is None:
        # Estimate max shape in the batch for padding
            shapes = []
            for image, _ in batch:
                shapes.append(image.shape)

            max_shape_for_padding = torch.max(torch.tensor(shapes), axis=0)[0]
        else:
            max_shape_for_padding = torch.tensor(self._max_shape)
        
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
        batch_bboxes, batch_classes = segmentation2bbox(torch.stack(batch_labels), self._bbox_padding)

        return torch.stack(batch_images), torch.stack(batch_masks), list(zip(batch_bboxes, batch_classes)), torch.stack(batch_labels)
