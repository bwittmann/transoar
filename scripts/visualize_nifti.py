"""Script to visualize data of the NIfTI format."""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from transoar.utils.io import load_nifti

def normalize(image):
    norm_img = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return norm_img

def show_images(images):
    for image in images:
        cv2.imshow('Raw NIfTI', image)
        cv2.waitKey()


if __name__ == "__main__":
    # PATH_TO_FILE = Path('/home/bastian/Datasets/CT_GC/10000006_1/10000006_1_CT_wb.nii.gz')
    # PATH_TO_FILE = Path('/home/bastian/Downloads/Task10_Colon/Task10_Colon/labelsTr/colon_011.nii.gz')
    # PATH_TO_FILE = Path('/home/bastian/Datasets/nndetection/Task000D3_Example/raw_splitted/labelsTr/case_7_0000.nii.gz')
    PATH_TO_FILE = Path('/home/bastian/Datasets/nndetection/Task101_OrganDet/raw_splitted/imagesTr/case_000_0000.nii.gz')
    

    # Load data from nifti
    data = load_nifti(PATH_TO_FILE)['data']
    print('Shape: ', data.shape)
    print('Labels: ', np.unique(data))

    # March through volume in all three directions
    images = defaultdict(list)

    for layer in [int(x) for x in np.linspace(0, data.shape[0] - 1 , 5)]:
        images['axis_0'].append(data[layer, :, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[1] - 1, 5)]:
        images['axis_1'].append(data[:, layer, :])

    for layer in [int(x) for x in np.linspace(0, data.shape[2] - 1, 5)]:
        images['axis_2'].append(data[:, :, layer])

    image_axis_0 = normalize(np.concatenate(images['axis_0'], axis=1))
    image_axis_1 = normalize(np.concatenate(images['axis_1'], axis=1))
    image_axis_2 = normalize(np.concatenate(images['axis_2'], axis=1))

    # show_images([image_axis_0, image_axis_1, image_axis_2])