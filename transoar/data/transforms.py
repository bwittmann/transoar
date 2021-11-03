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
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform
)
from batchgenerators.transforms.color_transforms import (
    GammaTransform,
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor


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


def get_transforms(split, data_config):
    if split == 'train':
        rotate_range = [i / 180 * np.pi for i in data_config['rotation']]
        transform = BGCompose(
            [
                SpatialTransform(
                    None, patch_center_dist_from_border=None, do_elastic_deform=False,
                    do_rotation=True, angle_x=rotate_range, angle_y=rotate_range,
                    angle_z=rotate_range, do_scale=True, scale=data_config['scale_range'],
                    order_data=3, border_mode_data='constant', border_cval_data=0, order_seg=0,
                    border_mode_seg='constant', border_cval_seg=0, random_crop=False,
                    p_scale_per_sample=data_config['p_scale'], p_rot_per_sample=data_config['p_rotation'],
                    independent_scale_for_each_axis=False, data_key='image', label_key='label'
                ),
                GaussianNoiseTransform(p_per_sample=0.1, data_key='image'),
                GaussianBlurTransform(
                    (0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                    p_per_channel=0.5, data_key='image'
                ),
                BrightnessMultiplicativeTransform(
                    multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key='image'
                ),
                ContrastAugmentationTransform(p_per_sample=0.15, data_key='image'),
                GammaTransform(
                    data_config['gamma_range'], True, True, retain_stats=True, p_per_sample=0.1,
                    data_key='image'
                ),
                GammaTransform(
                    data_config['gamma_range'], False, True, retain_stats=True, 
                    p_per_sample=data_config['p_gamma'], data_key='image'
                ),
                MirrorTransform(data_config['mirror_axes'], data_key='image', label_key='label'),
                NumpyToTensor(['image', 'label'], 'float')
            ]
        )
        return transform
    elif split == 'val':
        transform = BGCompose(
            [
                NumpyToTensor(['image', 'label'], 'float')
            ]
        )
        return transform
    elif split == 'test':
        transform = BGCompose(
            [
                NumpyToTensor(['image', 'label'], 'float')
            ]
        )
        return transform
