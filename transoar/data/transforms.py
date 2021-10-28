"""Transformations for different operations."""

from monai.transforms import (
    CropForegroundd,
)

def transform_crop(margin, key):
    transform = CropForegroundd(
        keys=['image', 'label'], source_key=key, margin=margin
    )
    return transform