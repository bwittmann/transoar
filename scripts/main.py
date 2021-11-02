"""Main script."""

from transoar.data.dataset import TransoarDataset
from transoar.utils.io import get_config, write_nifti
from transoar.utils.visualization import visualize_voxel_grid

import numpy as np

if __name__ == "__main__":
    split = 'train'
    test = TransoarDataset(get_config('data'), split)
    data = test[1]
    visualize_voxel_grid(data[0])
    write_nifti(data[0].squeeze(), {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/data.nii.gz')
    write_nifti(data[1].squeeze(), {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/label.nii.gz')