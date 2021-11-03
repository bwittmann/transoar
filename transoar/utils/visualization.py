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

def convert_bboxes(labels, data=None, seg_map=None, standalone=True, value=None):
    """Converts bounding boxes to a 3D volume.

    This makes it possible to visualize bboxes of the format x1, y1, z1, x2, y2, z2
    with a tool like ITK-SNAP.
    
    Args:
        labels: A tuple consisting of a torch.tensor of the shape [N, 6] representing bboxes
            and a list of length N representing the respective classes.
        data: A torch.tensor representing the the image data.
        seg_map: A torch.tensor representing the segmentation labels.
        standalone: If True, add bounding boxes to a np.ndarray full of zeros.
        value: If not None, the bboxes will have this value in the array.

    Returns:
        A np.ndarray containing the bboxes in a visualizable format.
    """
    # Generate volume to add bboxes
    if standalone:
        if data is not None:
            bboxes_volume = np.zeros_like(data).squeeze()
        elif seg_map is not None:
            bboxes_volume = np.zeros_like(seg_map).squeeze()
        else:
            raise RuntimeError('Please input either the data or the seg_map.')
    elif data is not None:
        bboxes_volume = data.numpy().squeeze()
    elif seg_map is not None:
        bboxes_volume = seg_map.numpy().squeeze()

    bboxes = labels[0].split(1)
    classes = labels[1]
    for bbox, class_ in zip(bboxes, classes):
        x1, y1, z1, x2, y2, z2 = bbox[0].tolist()

        if value:
            bbox_val = value
        else:
            bbox_val = class_

        for y, z in [(y1, z1), (y1, z2), (y2, z1), (y2, z2)]:
            for x_val in range(x1, x2 + 1):
                bboxes_volume[x_val, y, z] = bbox_val

        for x, z in [(x1, z1), (x1, z2), (x2, z1), (x2, z2)]:
            for y_val in range(y1, y2 + 1):
                bboxes_volume[x, y_val, z] = bbox_val

        for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
            for z_val in range(z1, z2 + 1):
                bboxes_volume[x, y, z_val] = bbox_val

    return bboxes_volume
        

            



        
    
    

