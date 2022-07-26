"""Module to analyze properties of the amos dataset and preprocess raw cases."""

import os
import logging
from collections import defaultdict

import torch
import numpy as np

from transoar.data.transforms import transform_preprocessing_amos
from transoar.utils.io import write_json
from transoar.utils.bboxes import segmentation2bbox, box_cxcyczwhd_to_xyzxyz

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
        path_to_dataset,
        path_to_splits,
        preprocessing_config,
        data_config
    ):
        self._preprocessing_config = preprocessing_config
        self._data_config = data_config

        self._preprocessing_transform = transform_preprocessing_amos(
            margin=preprocessing_config['margin'],
            crop_key=preprocessing_config['key'], 
            orientation=preprocessing_config['orientation'],
            resize_shape=preprocessing_config['resize_shape']
        )

        self._path_to_dataset = path_to_dataset
        self._path_to_splits = path_to_splits
        self._splits = {
            'train': paths_to_train,
            'val': paths_to_val ,
            'test': paths_to_test
        }

        self._shapes = []
        self._bboxes = []
        self._norm_voxels = []

    def run(self):
        for split_name, split_paths in self._splits.items():
            logging.info(f'Preparing {split_name} set.')
            for idx, case in enumerate(split_paths):
                path_image, path_label = self._path_to_dataset / case['image'], self._path_to_dataset / case['label']
                case_name = case['image'].split('/')[-1][5:9]

                case_dict = {
                        'image': path_image,
                        'label': path_label
                    }

                preprocessed_case = self._preprocessing_transform(case_dict)
                image, label = preprocessed_case['image'], preprocessed_case['label']
                assert image.shape == label.shape

                # skip cases that dont contain important border organs for cropping
                unique_labels = np.unique(label)
                if unique_labels.shape[0] != 16:
                    contains_border = all([m  in unique_labels.tolist() for m in [15., 14., 6., 1., 7.]])
                    if contains_border == False:
                        logging.info(f"Skipped case {case_name} due to missing border organs.")
                        continue

                # check boundary organs in fov
                margin_boundary = 1
                boundaries = [
                    label[0, 0:margin_boundary, :, :],
                    label[0, :, 0:margin_boundary, :],
                    label[0, :, :, 0:margin_boundary],
                    label[0, -margin_boundary:, :, :],
                    label[0, :, -margin_boundary:, :],
                    label[0, :, :, -margin_boundary:],
                ]
                crossed_boundary = False
                for boundary in boundaries:
                    for border_org in [15., 14., 6., 1., 7.]:
                        if border_org in boundary:
                            crossed_boundary = True

                if crossed_boundary == True:
                    logging.info(f"Skipped case {case_name} due to crossed boundary.")
                    continue

                if split_name != 'test':
                    self._shapes.append(image.shape)

                    bboxes, classes = segmentation2bbox(torch.tensor(label[None, ...]), padding=1)
                    self._bboxes.append([bboxes, classes])

                    voxels_foreground = self._get_foreground_voxels(image, label)
                    self._norm_voxels += voxels_foreground

                logging.info(f'Successfull prepared case {case_name} of shape {image.shape}.')

                path_to_case = self._path_to_splits / split_name / case_name
                
                os.makedirs(path_to_case)

                np.save(str(path_to_case / 'data.npy'), image.astype(np.float32))
                np.save(str(path_to_case / 'label.npy'), label.astype(np.int32))

        self._data_config['bbox_properties'] = self._get_bbox_props()
        self._data_config['shape_statistics'] = self._get_shape_statistics()
        self._data_config['foreground_voxel_statistics'] = self._get_voxel_statistics()
        self._data_config['preprocessing_config'] = self._preprocessing_config

        # Save relevant information of dataset and preprocessing
        write_json(self._data_config, self._path_to_splits / 'data_info.json')

    def _get_bbox_props(self):
        bbox_dict = defaultdict(list)
        bbox_properties = {}

        for bboxes, classes in self._bboxes:
            for bbox, class_ in zip(bboxes[0], classes[0]):
                bbox_dict[class_.item()].append(bbox)

        for class_ in bbox_dict.keys():
            class_bboxes = torch.vstack(bbox_dict[class_]).numpy()

            # Get general information about position of bboxes
            bbox_properties[class_] = {
                "median": np.median(class_bboxes, axis=0).tolist(), # cx, cy, cz, w, h, d
                "mean": np.mean(class_bboxes, axis=0).tolist(),
                "min": np.min(class_bboxes, axis=0).tolist(),
                "max": np.max(class_bboxes, axis=0).tolist(),
                "percentile_99_5": np.percentile(class_bboxes, 99.5, axis=0).tolist(),
                "percentile_00_5": np.percentile(class_bboxes, 0.5, axis=0).tolist()
            }

            # Get the area to apply attn to
            min_pos = np.min(box_cxcyczwhd_to_xyzxyz(class_bboxes), axis=0)
            max_pos = np.max(box_cxcyczwhd_to_xyzxyz(class_bboxes), axis=0)

            attn_area = [   # x1, y1, z1, x2, y2, z2
                min_pos[0].item(),
                min_pos[1].item(),
                min_pos[2].item(),
                max_pos[3].item(),
                max_pos[4].item(),
                max_pos[5].item()
            ]
            bbox_properties[class_]['attn_area'] = attn_area

        return bbox_properties

    def _get_foreground_voxels(self, data, seg, subsample=10):
        mask = seg > 0
        return list(data[mask.astype(bool)][::subsample])
    
    def _get_shape_statistics(self):
        shapes = np.array(self._shapes, dtype=np.int)[:, 1:]
        shape_statistics = {
            "median": np.median(shapes, axis=0).astype(np.int).tolist(),
            "mean": np.mean(shapes, axis=0).tolist(),
            "min": np.min(shapes, axis=0).tolist(),
            "max": np.max(shapes, axis=0).tolist(),
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
