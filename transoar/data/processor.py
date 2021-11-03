"""Module to pre-process raw cases."""

import logging
import os

import numpy as np

from transoar.data.transforms import transform_preprocessing


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
        min_num_organs = self._data_config['min_num_organs']

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

                # Skip cases with a small amount of labels
                if np.unique(label).size < min_num_organs:
                    logging.info(f'Skipped case {case.name} with less than {min_num_organs} organs.')
                    continue

                logging.info(f'Successfull prepared case {case.name} of shape {image.shape}.')

                path_to_case = self._path_to_splits / split_name / case.name
                os.makedirs(path_to_case)
                np.save(str(path_to_case / 'data.npy'), image.astype(np.float32))
                np.save(str(path_to_case / 'label.npy'), label.astype(np.int32))
