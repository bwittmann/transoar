"""Script for training."""

import numpy as np
import torch

from transoar.data.dataloader import get_loader
from transoar.utils.io import get_complete_config, write_nifti
from transoar.utils.visualization import convert_bboxes


def train(config):
    loader = get_loader(config['data'], 'train')

    for data, mask, bboxes, seg_labels in loader:
        # ret = convert_bboxes(labels=bboxes[0], seg_map=seg_labels[0], standalone=True)
        # write_nifti(ret, {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/ret.nii.gz')

        # write_nifti(data[0][0], {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/data.nii.gz')
        # write_nifti(seg_labels[0][0], {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/labels.nii.gz')
        # write_nifti(mask[0][0], {'itk_spacing': [1, 1, 1]}, '/home/bastian/Downloads/mask.nii.gz')
        pass


if __name__ == "__main__":
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10)

    config = get_complete_config()

    train(config)
      