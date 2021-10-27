"""Module to analyze properties of the dataset."""

import logging

import numpy as np
from tqdm import tqdm 

from transoar.utils.bboxes import segmentation2bbox, iou_3d
from transoar.utils.io import load_case


logging.basicConfig(level=logging.INFO)

class DataSetAnalyzer:
    """Analyzer to analyze properties of dataset."""

    def __init__(self, paths_to_cases, data_config):
        self._paths_to_cases = paths_to_cases
        self._data_config = data_config

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
   
        logging.info('Calculating properties based on analysis of dataset.')
        voxel_statistics = self._get_voxel_statistics()
        self._class_ious = np.concatenate(self._class_ious)

        if self._data_config['target_spacing']:
            target_spacing = self._data_config['target_spacing']
        else:
            target_spacing = self._get_target_spacing()

        median_shape_new, min_shape_new, max_shape_new = self._determine_new_shapes(target_spacing)
        class_weights = self._get_class_weights()
        anchors = self._get_anchors()

        ret_dict = {
            'statistics': voxel_statistics,
            'all_ious': self._class_ious,
            'num_instances': self._num_instances,
            'volume_per_class': self._volume_per_class,
            'shapes': self._shapes,
            'spacing': self._spacings,
            'target_spacing': target_spacing,
            'median_shape_new': median_shape_new,
            'min_shape_new': min_shape_new,
            'max_shape_new': max_shape_new,
            'class_weights': class_weights,
            'anchors': anchors
        }

        return ret_dict

    def _get_anchors(self):
        pass

    def _get_class_weights(self):
        """
        background weight: 1 / (num_classes + 1)
        foreground_weight: (1 - 1 / (num_classes + 1))*(1 - ni / nall)
        """

        num_classes = len(self._data_config['labels'].keys())
        weight_background = 1 / (num_classes + 1)
        remaining_weight = 1 - weight_background

        num_all_instances = sum(self._num_instances.values())

        weight_classes = []
        for num_instances in self._num_instances.values():
            weight_class = remaining_weight * (1 - num_instances / num_all_instances)
            weight_classes.append(weight_class)

        return weight_background, *weight_classes

    def _determine_new_shapes(self, target_spacing):
        new_shapes = []
        for spacing, shape in zip(self._spacings, self._shapes):
            new_shape = np.array(spacing) / target_spacing * np.array(shape)
            new_shapes.append(new_shape)

        new_shapes = np.vstack(new_shapes)
        median_shape_new = np.round(np.median(new_shapes, 0))
        max_shape_new = np.ceil(np.max(new_shapes, 0))
        min_shape_new = np.floor(np.min(new_shapes, 0))

        return median_shape_new, min_shape_new, max_shape_new

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

        
