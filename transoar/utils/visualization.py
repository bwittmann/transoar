"""Helper functions for visualization purposes."""

from collections import defaultdict

import cv2
import numpy as np


def show_images(images):
    """Displays a series of images.
    
    Args: A list containing arbitrary many 2D np.ndarrays
    """
    for image in images:
        cv2.imshow('Raw NIfTI', image)
        cv2.waitKey()

def normalize(image):
    """Normalizes an image to uint8 to display as greyscale.

    Args:
        image: A np.ndarray of dim 3d representing data or our modality.

    Returns:
        The same image, but converted to uint8.
    """
    norm_img = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return norm_img

def visualize_voxel_grid(data):
    """Visualizes layers of a voxel grid.

    Args:
        data: A np.ndarray corresponding to the voxel grid data.
    """
    images = defaultdict(list)

    data = normalize(data.squeeze())
    assert len(data.shape) == 3, 'Data has to be 3D.'

    for layer in [int(x) for x in np.linspace(0, data.shape[0] - 1 , 5)]:
        images['axis_0'].append(data[layer, :, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[1] - 1, 5)]:
        images['axis_1'].append(data[:, layer, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[2] - 1, 5)]:
        images['axis_2'].append(data[:, :, layer])

    image_axis_0 = normalize(np.concatenate(images['axis_0'], axis=1))
    image_axis_1 = normalize(np.concatenate(images['axis_1'], axis=1))
    image_axis_2 = normalize(np.concatenate(images['axis_2'], axis=1))

    show_images([image_axis_0, image_axis_1, image_axis_2])
