"""Transformations for different operations."""

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    MapTransform
)


def transform_crop(margin, crop_key):
    transform = CropForegroundd(
        keys=['image', 'label'], source_key=crop_key, margin=margin
    )
    return transform

def transform_preprocessing(
    margin, crop_key, orientation, target_spacing, clip_min, 
    clip_max, std, mean
):
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"], pixdim=target_spacing,
                 mode=("bilinear", "nearest")
            ),
            Orientationd(keys=["image", "label"], axcodes=orientation),
            NormalizeClipd(
                keys=['image'], clip_min=clip_min, clip_max=clip_max, 
                std=std, mean=mean
            ),
            CropForegroundd(
                keys=["image", "label"], source_key=crop_key, 
                margin=margin
            )
        ]
    )
    return transform

class NormalizeClipd(MapTransform):
    def __init__(self, keys, clip_min, clip_max, std, mean):
        self._key = keys
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._std = std
        self._mean = mean

    def __call__(self, data):
        key = self._key[0]
        data[key] = np.clip(data[key], self._clip_min, self._clip_max)
        data[key] = (data[key] - self._mean) / self._std
        return data
