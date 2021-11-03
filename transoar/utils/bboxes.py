"""Helper functions for handling bounding boxes."""

import torch
import numpy as np


def segmentation2bbox(segmentation_maps, padding):
    batch_bboxes = []
    batch_classes = []
    for map_ in segmentation_maps:
        assert map_.ndim == 4

        bboxes = []
        classes = [int(class_) for class_ in map_.unique(sorted=True)][1:]
        batch_classes.append(classes)

        for class_ in classes:
            class_indices = (map_ == class_).nonzero(as_tuple=False)

            min_values = class_indices.min(dim=0)[0][1:]    # x, y, z
            max_values = class_indices.max(dim=0)[0][1:]

            # Apply padding to bounding boxes
            min_values -= padding
            max_values += padding

            assert min_values[0] < max_values[0]
            assert min_values[1] < max_values[1]
            assert min_values[2] < max_values[2]

            bboxes.append(torch.hstack((min_values, max_values)))   # x1, y1, z1, x2, y2, z2
        batch_bboxes.append(torch.vstack(bboxes))

    return batch_bboxes, batch_classes

def iou_3d(bboxes1, bboxes2):
    """Determines the intersection over union (IoU) for two sets of
    three dimensional bounding boxes.

    Bounding boxes have to be in the format (x1, y1, z1, x2, y2, z2).

    Args:
        bboxes1: A np.ndarray of the shape [N, 6] containing the first
            set of bounding boxes.
        bboxes2: A np.ndarray of the shape [M, 6] containing the first
            set of bounding boxes.

    Returns:
        A np.ndarray of shape [N, M] containing the IoU values of all 
        bounding boxes to each other. Keep in mind that the diagonal
        consists of ones.
    """
    volume_bbox1 = bboxes_volume(bboxes1)
    volume_bbox2 = bboxes_volume(bboxes2)

    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    z1 = np.maximum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    x2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])
    y2 = np.minimum(bboxes1[:, None, 4], bboxes2[None, :, 4])
    z2 = np.minimum(bboxes1[:, None, 5], bboxes2[None, :, 5])

    delta_x = np.clip((x2 - x1), a_min=0, a_max=None)
    delta_y = np.clip((y2 - y1), a_min=0, a_max=None)
    delta_z = np.clip((z2 - z1), a_min=0, a_max=None)

    intersection = delta_x * delta_y * delta_z
    union = volume_bbox1[:, None] + volume_bbox2 - intersection

    return intersection / union

def bboxes_volume(bboxes):
    """Estimates the volume of a three dimensional bounding box.
    
    Args:
        bboxes: A np.ndarray of the shape [N, 6] containing N bounding
            boxes in the format (x1, y1, z1, x2, y2, z2).

    Returns:
        A np.ndarray of shape (N,) containing the corresponding volumes.
    """
    delta_x = bboxes[:, 3] - bboxes[:, 0]
    delta_y = bboxes[:, 4] - bboxes[:, 1]
    delta_z = bboxes[:, 5] - bboxes[:, 2]
    return delta_x * delta_y * delta_z