"""Helper functions for handling bounding boxes."""

import numpy as np


def segmentation2bbox(loaded_case, padding):
    """Converts a segmentation map to a list of bounding boxes.
    
    Args:
        loaded_case: A dict containing the data and meta data of a loaded case
            To load a case please refer to transoar/utils/io.py.
        padding: An integer that allows to adust the padding of the bounding boxes.

    Returns:
        A list of dicts. Each dict describes a specific bounding box, where the key
        'bbox' refers to the (x1, y1, z1, x2, y2, z2) component and the key 'label' to the
        corresponding class.
    """
    bboxes = []
    for class_ in loaded_case['meta_data']['classes']:
        class_indices = np.argwhere(loaded_case['data'][1] == class_)

        min_values = np.min(class_indices, axis=0)  # x, y, z
        max_values = np.max(class_indices, axis=0)

        # Apply padding to bounding boxes
        min_values -= padding
        max_values += padding

        bbox = {
            'bbox': np.hstack((min_values, max_values)),
            'label': class_
        }

        bboxes.append(bbox)

    return bboxes




