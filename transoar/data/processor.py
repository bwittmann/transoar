"""Module to pre-process raw cases."""

import logging
import os

import numpy as np

from transoar.data.transforms import transform_preprocessing
from transoar.utils.bboxes import segmentation2bbox
from transoar.utils.io import write_pkl, write_nifti
from transoar.utils.visualization import visualize_voxel_grid


logging.basicConfig(level=logging.INFO)

class Preprocessor:
    """Preprocessor to pre-process raw data samples."""

    def __init__(
        self, paths_to_train, paths_to_val, paths_to_test, data_config, 
        path_to_splits, analysis
    ):
        self._data_config = data_config
        self._path_to_splits = path_to_splits
        self._analysis = analysis

        self._preprocessing_transform = transform_preprocessing(
            margin=data_config['margin'], crop_key=data_config['key'], orientation=data_config['orientation'],
            target_spacing=analysis['target_spacing'][[2, 1, 0]], clip_min=analysis['statistics']['percentile_00_5'],
            clip_max=analysis['statistics']['percentile_99_5'], std=analysis['statistics']['std'],
            mean=analysis['statistics']['mean'],
        )

        self._splits = {
            'train': paths_to_train,
            'val': paths_to_val ,
            'test': paths_to_test
        }


    def prepare_sets(self):
        for split_name, split_paths in self._splits.items():
            logging.info(f'Preparing {split_name} set.')
            for case in (split_paths):
                path_image, path_label = sorted(list(case.iterdir()), key=lambda x: len(str(x)))

                case_dict = {
                    'image': path_image,
                    'label': path_label
                }

                preprocessed_case = self._preprocessing_transform(case_dict)
                image, label = preprocessed_case['image'], preprocessed_case['label']
                # bboxes = segmentation2bbox(label, self._data_config['bbox_padding'])

                # visualize_voxel_grid(image)
                # visualize_voxel_grid(label)
                
                logging.info(f'Successfull prepared case {case.name} of shape {image.shape}.')

                path_to_case = self._path_to_splits / split_name / case.name
                os.makedirs(path_to_case)
                # write_nifti(image.squeeze(), {'itk_spacing': [1, 1, 1]}, str(path_to_case / 'data.nii.gz'))
                # write_nifti(label.squeeze(), {'itk_spacing': [1, 1, 1]}, str(path_to_case / 'label.nii.gz'))
                np.save(str(path_to_case / 'data.npy'), image.astype(np.float32))
                np.save(str(path_to_case / 'label.npy'), label.astype(np.int32))
                # write_pkl(bboxes, path_to_case / 'bboxes.pkl')
