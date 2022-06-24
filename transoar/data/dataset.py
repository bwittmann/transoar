"""Module containing the dataset related functionality."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from transoar.data.transforms import get_transforms


class TransoarDataset(Dataset):
    """Dataset class of the transoar project."""
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self._config = config

        data_dir = Path("./dataset/").resolve()
        self._path_to_split = data_dir / self._config['dataset'] / split
        self._data = [data_path.name for data_path in self._path_to_split.iterdir()]

        # if split == 'train':
        #     self._data = self._data[:12]
        #     self._data = self._data[:60]

        self._augmentation = get_transforms(split, config)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._config['overfit']:
            idx = 0

        case = self._data[idx]
        path_to_case = self._path_to_split / case
        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)

        if self._config['augmentation']['use_augmentation']:
            data_dict = {
                'image': data,
                'label': label
            }

            # Apply data augmentation
            self._augmentation.set_random_state(torch.initial_seed() + idx)

            data_transformed = self._augmentation(data_dict)
            data, label = data_transformed['image'], data_transformed['label']
        else:
            data, label = torch.tensor(data), torch.tensor(label)

        return data, label
