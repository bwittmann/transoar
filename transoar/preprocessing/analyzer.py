"""Module to analyze properties of the dataset."""

import logging

import numpy as np
from tqdm import tqdm 

from transoar.utils.bboxes import segmentation2bbox, iou_3d
from transoar.utils.io import load_case, get_config


logging.basicConfig(level=logging.INFO)

class DataSetAnalyzer:
    """Analyzer to analyze properties of dataset."""

    def __init__(self, paths_to_cases):
        self._paths_to_cases = paths_to_cases
        self._data_config = get_config('data')

        # Init structures to collect properties
        self._shapes = []
        self._spacings = []
        self._foreground_voxels = []
        self._num_instances = {class_id: 0 for class_id in self._data_config['labels']}
        self._volume_per_class = {class_id: 0 for class_id in self._data_config['labels']}
        self._class_ious = []

    def analyze(self):
        logging.info('Analyze dataset properties.')
        # Loop over cases and determine properties
        for case in tqdm(self._paths_to_cases):
            loaded_case = load_case(list(case.iterdir()))
            if loaded_case == None:
                continue

            self._shapes.append(loaded_case['data'].shape[1:])
            self._spacings.append(loaded_case['meta_data']['original_spacing'])

            # Get voxels from foreground
            voxels_foreground = self._get_foreground_voxels(loaded_case)
            self._foreground_voxels += voxels_foreground

            # Check if classes are present in the current case
            for class_ in loaded_case['meta_data']['classes']:
                self._num_instances[class_] += 1

            # Estimate volumes for each class
            self._update_volumes(loaded_case)

            # Estimate iou values of all bboxes in current case
            bboxes = segmentation2bbox(loaded_case, self._data_config['bbox_padding'])
            class_ious = self._determine_iou(bboxes)
            self._class_ious.append(class_ious.flatten())

            
        voxel_statistics = self._get_voxel_statistics()
        self._class_ious = np.concatenate(self._class_ious)

        ret_dict = {
            'statistics': voxel_statistics,
            'all_ious': class_ious,
            'num_instances': self._num_instances,
            'volume_per_class': self._volume_per_class,
            'shapes': self._shapes,
            'spacing': self._spacings
        }

        return ret_dict

        
    def _get_foreground_voxels(self, loaded_case, subsample=10):
        data, seg = loaded_case['data'][0], loaded_case['data'][1]
        mask = seg > 0
        return list(data[mask.astype(bool)][::subsample])

    def _get_voxel_statistics(self):
        voxel_statistics = {
            "median": np.median(self._foreground_voxels),
            "mean": np.mean(self._foreground_voxels),
            "std": np.std(self._foreground_voxels),
            "min": np.min(self._foreground_voxels),
            "max": np.max(self._foreground_voxels),
            "percentile_99_5": np.percentile(self._foreground_voxels, 99.5),
            "percentile_00_5": np.percentile(self._foreground_voxels, 00.5),
        }
        return voxel_statistics

    def _update_volumes(self, loaded_case):
        voxel_volume = np.prod(loaded_case['meta_data']['itk_spacing']) # unit: mm^3

        for class_ in loaded_case['meta_data']['classes']:
            class_volume = np.sum(loaded_case['data'][1] == class_) * voxel_volume
            self._volume_per_class[class_] += np.round(class_volume, 2)

    def _determine_iou(self, bboxes):
        bboxes = np.vstack([bbox['bbox'] for bbox in bboxes])
        class_ious = iou_3d(bboxes, bboxes)

        # Get rid of diagonal which are always 1
        return class_ious[~np.eye(class_ious.shape[0], dtype=bool)].reshape(class_ious.shape[0], -1)

        
