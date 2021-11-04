"""Module to analyze properties of the dataset."""

import logging

import numpy as np
from tqdm import tqdm

from transoar.data.transforms import transform_crop
from transoar.utils.io import load_case


class DataSetAnalyzer:
    """Analyzer to analyze properties of dataset."""
    def __init__(self, paths_to_cases, data_config):
        self._paths_to_cases = paths_to_cases
        self._data_config = data_config

        # Init structures to collect properties
        self._shapes = []
        self._spacings = []
        self._norm_voxels = []

        self._cropper = transform_crop(
            data_config['margin'],
            data_config['key']
        )

    def analyze(self):
        logging.info('Analyze dataset properties.')
        # Loop over cases and determine properties
        for case in tqdm(self._paths_to_cases):
            loaded_case = load_case(list(case.iterdir()))
            if loaded_case == None:
                continue

            if self._data_config['cropping']:
                case_dict = {
                    'image': loaded_case['data'][0][None],
                    'label': loaded_case['data'][1][None]
                }

                case_cropped = self._cropper(case_dict)
                loaded_case['data'] = np.concatenate((case_cropped['image'], case_cropped['label']))

            if self._data_config['foreground_normalization']:
                voxels_foreground = self._get_foreground_voxels(loaded_case)
                self._norm_voxels += voxels_foreground
            else:
                voxels_sparse = self._get_voxels(loaded_case)
                self._norm_voxels += voxels_sparse

            self._shapes.append(loaded_case['data'].shape[1:])
            self._spacings.append(loaded_case['meta_data']['original_spacing'])

        logging.info('Calculating properties based on analysis of dataset.')
        voxel_statistics = self._get_voxel_statistics()

        if self._data_config['target_spacing']:
            target_spacing = np.array(self._data_config['target_spacing'])
        else:
            target_spacing = self._get_target_spacing()

        ret_dict = {
            'statistics': voxel_statistics, 
            'shapes': self._shapes,
            'spacing': self._spacings,
            'target_spacing': target_spacing
        }

        return ret_dict

    def _get_target_spacing(self):
        """Adapted from nndet"""
        target_spacing = np.percentile(np.vstack(self._spacings), self._data_config['target_spacing_percentile'], 0)
        target_shape = np.percentile(np.vstack(self._shapes), self._data_config['target_spacing_percentile'], 0)

        worst_spacing_axis = np.argmax(target_spacing)
        other_axes = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
        other_spacings = [target_spacing[i] for i in other_axes]
        other_sizes = [target_shape[i] for i in other_axes]

        has_aniso_spacing = target_spacing[worst_spacing_axis] > (self._data_config['anisotropy_threshold'] * min(other_spacings))
        has_aniso_voxels = target_shape[worst_spacing_axis] * self._data_config['anisotropy_threshold'] < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(self._spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            if target_spacing_of_that_axis < min(other_spacings):
                target_spacing_of_that_axis = max(min(other_spacings), target_spacing_of_that_axis) + 1e-5
            target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

        return target_spacing

    def _get_foreground_voxels(self, loaded_case, subsample=10):
        data, seg = loaded_case['data'][0], loaded_case['data'][1]
        mask = seg > 0
        return list(data[mask.astype(bool)][::subsample])

    def _get_voxels(self, loaded_case, subsample=200):
        data = loaded_case['data'][0]
        return list(data.flatten())[::subsample]

    def _get_voxel_statistics(self):
        voxel_statistics = {
            "median": np.median(self._norm_voxels),
            "mean": np.mean(self._norm_voxels),
            "std": np.std(self._norm_voxels),
            "min": np.min(self._norm_voxels),
            "max": np.max(self._norm_voxels),
            "percentile_99_5": np.percentile(self._norm_voxels, 99.5),
            "percentile_00_5": np.percentile(self._norm_voxels, 0.5),
        }
        return voxel_statistics
