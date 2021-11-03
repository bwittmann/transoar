"""Main script."""

import numpy as np
import torch

from transoar.data.dataset import TransoarDataset
from transoar.data.dataloader import get_loader
from transoar.utils.io import get_config, write_nifti
from transoar.utils.visualization import visualize_voxel_grid


if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    split = 'train'
    # test = TransoarDataset(get_config('data'), split)
    # data = test[1]
    # visualize_voxel_grid(data[0])
    # write_nifti(data[0].squeeze(), {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/data.nii.gz')
    # write_nifti(data[1].squeeze(), {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/label.nii.gz')

    loader = get_loader(get_config('data'), split)

    for data, mask, labels in loader:
        write_nifti(data[0].squeeze(), {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/data.nii.gz')
        k = 12
        