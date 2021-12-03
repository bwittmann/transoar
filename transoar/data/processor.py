"""Module to pre-process raw cases."""

import logging
import os

import numpy as np

from transoar.data.transforms import transform_preprocessing
from transoar.utils.io import write_json


class Preprocessor:
    """Preprocessor to pre-process raw data samples."""
    def __init__(
        self, paths_to_train, paths_to_val, paths_to_test, preprocessing_config, data_config,
        path_to_splits, analysis
    ):
        self._preprocessing_config = preprocessing_config
        self._data_config = data_config
        self._path_to_splits = path_to_splits
        self._analysis = analysis

        self._preprocessing_transform = transform_preprocessing(
            margin=preprocessing_config['margin'], crop_key=preprocessing_config['key'], orientation=preprocessing_config['orientation'],
            target_spacing=analysis['target_spacing'][[2, 1, 0]], clip_min=analysis['voxel_statistics']['percentile_00_5'],
            clip_max=analysis['voxel_statistics']['percentile_99_5'], std=analysis['voxel_statistics']['std'],
            mean=analysis['voxel_statistics']['mean']
        )

        self._splits = {
            'train': paths_to_train,
            'val': paths_to_val ,
            'test': paths_to_test
        }

    def prepare_sets(self):
        shapes = []
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
                shapes.append(image.shape)

                # Skip cases with a small amount of labels
                if np.unique(label).size < self._preprocessing_config['min_num_organs']:
                    logging.info(f"Skipped case {case.name} with less than {self._preprocessing_config['min_num_organs']} organs.")
                    continue

                logging.info(f'Successfull prepared case {case.name} of shape {image.shape}.')

                path_to_case = self._path_to_splits / split_name / case.name
                os.makedirs(path_to_case)
                np.save(str(path_to_case / 'data.npy'), image.astype(np.float32))
                np.save(str(path_to_case / 'label.npy'), label.astype(np.int32))
            
        if self._preprocessing_config['fixed_size']:
            dict_to_save = {
                'max_size': np.max(np.array(shapes), axis=0).tolist()
            }
            self._data_config.update(dict_to_save)

        # Save relevant information of dataset and preprocessing
        write_json(self._data_config, self._path_to_splits / 'data_info.json')