"""Module to analyze properties of the dataset and preprocess raw cases."""

import os
import logging

import numpy as np

from transoar.data.transforms import transform_preprocessing
from transoar.utils.io import write_json

class PreProcessor:
    """Data preprocessor of the transoar project.
    
    Analyzes and extracts necessary properties of the dataset and preprocesses
    and saves raw cases as .npy files.
    """
    def __init__(
        self,
        paths_to_train,
        paths_to_val,
        paths_to_test,
        path_to_splits,
        preprocessing_config,
        data_config
    ):
        self._preprocessing_config = preprocessing_config
        self._data_config = data_config

        self._preprocessing_transform = transform_preprocessing(
            margin=preprocessing_config['margin'],
            crop_key=preprocessing_config['key'], 
            orientation=preprocessing_config['orientation'],
            target_spacing=preprocessing_config['target_spacing']
        )

        self._path_to_splits = path_to_splits
        self._splits = {
            'train': paths_to_train,
            'val': paths_to_val ,
            'test': paths_to_test
        }

        self._shapes = []
        self._norm_voxels = []

    def run(self):
        for split_name, split_paths in self._splits.items():
            logging.info(f'Preparing {split_name} set.')
            for case in split_paths:
                path_image, path_label = sorted(list(case.iterdir()), key=lambda x: len(str(x)))

                case_dict = {
                        'image': path_image,
                        'label': path_label
                    }

                preprocessed_case = self._preprocessing_transform(case_dict)
                image, label = preprocessed_case['image'], preprocessed_case['label']

                if split_name != 'test':
                    self._shapes.append(image.shape)

                    voxels_foreground = self._get_foreground_voxels(image, label)
                    self._norm_voxels += voxels_foreground

                # Skip cases with a small amount of labels
                if np.unique(label).size < self._preprocessing_config['min_num_organs'] + 1:
                    logging.info(f"Skipped case {case.name} with less than {self._preprocessing_config['min_num_organs']} organs.")
                    continue

                logging.info(f'Successfull prepared case {case.name} of shape {image.shape}.')

                path_to_case = self._path_to_splits / split_name / case.name
                os.makedirs(path_to_case)

                np.save(str(path_to_case / 'data.npy'), image.astype(np.float32))
                np.save(str(path_to_case / 'label.npy'), label.astype(np.int32))

        self._data_config['shape_statistics'] = self._get_shape_statistics()
        self._data_config['foreground_voxel_statistics'] = self._get_voxel_statistics()
        self._data_config['preprocessing_config'] = self._preprocessing_config

        # Save relevant information of dataset and preprocessing
        write_json(self._data_config, self._path_to_splits / 'data_info.json')


    def _get_foreground_voxels(self, data, seg, subsample=10):
        mask = seg > 0
        return list(data[mask.astype(bool)][::subsample])
    
    def _get_shape_statistics(self):
        shapes = np.array(self._shapes, dtype=np.int)[:, 1:]
        shape_statistics = {
            "median": np.median(shapes, axis=0, dtype=np.int).tolist(),
            "mean": np.mean(shapes, axis=0, dtype=np.int).tolist(),
            "min": np.min(shapes, axis=0, dtype=np.int).tolist(),
            "max": np.max(shapes, axis=0, dtype=np.int).tolist(),
            "percentile_99_5": np.percentile(shapes, 99.5, axis=0).tolist(),
            "percentile_00_5": np.percentile(shapes, 0.5, axis=0).tolist()
        }
        return shape_statistics

    def _get_voxel_statistics(self):
        norm_voxels = np.array(self._norm_voxels, dtype=np.float)
        voxel_statistics = {
            "median": np.median(norm_voxels),
            "mean": np.mean(norm_voxels),
            "std": np.std(norm_voxels),
            "min": np.min(norm_voxels),
            "max": np.max(norm_voxels),
            "percentile_99_5": np.percentile(norm_voxels, 99.5),
            "percentile_00_5": np.percentile(norm_voxels, 0.5),
        }
        return voxel_statistics
