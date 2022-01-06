"""Transformations for different operations."""

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    MapTransform,
    ScaleIntensityRanged,
    Resized,
    RandSpatialCropd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandRotated,
    RandZoomd,
    RandAffined,
    RandFlipd,
    ToTensord
)


def crop_air(x):
    # To not crop fat which is -120 to -90
    return x > -500

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

def transform_preprocessing(
    margin, crop_key, orientation, target_spacing
):
    transform_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"], pixdim=target_spacing,
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes=orientation),
        CropForegroundd(
            keys=["image", "label"], source_key=crop_key, 
            margin=margin, select_fn=crop_air
        )
    ]

    return Compose(transform_list)

def get_transforms(split, config):
    rotate_range = [i / 180 * np.pi for i in config['augmentation']['rotation']]
    translate_range = [(i * config['augmentation']['translate_precentage']) / 100 for i in config['shape_statistics']['median']]
    
    if config['augmentation']['patch_size'] is None:
        patch_size = [int(x) for x in config['shape_statistics']['median']]
    else:
        patch_size = config['augmentation']['patch_size']

    if split == 'train':
        transform = [
            ScaleIntensityRanged(
                keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                # keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                # a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),

            # Spatial transformations
            Resized(        # Resize
                keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
                mode=['area', 'nearest']
            ),
            RandRotated(    # Rotation    
                keys=['image', 'label'], prob=config['augmentation']['p_rotate'],
                range_x=rotate_range, range_y=rotate_range, range_z=rotate_range,
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandZoomd(      # Zoom
                keys=['image', 'label'], prob=config['augmentation']['p_zoom'],
                min_zoom=config['augmentation']['min_zoom'],
                max_zoom=config['augmentation']['max_zoom'],
                mode=['area', 'nearest'], padding_mode='constant', constant_values=0
            ),
            RandAffined(    # Translation
                keys=['image', 'label'], prob=config['augmentation']['p_translate'],
                mode=['bilinear', 'nearest'],
                translate_range=translate_range, padding_mode='zeros'
            ), 
            RandAffined(    # Shear
                keys=['image', 'label'], prob=config['augmentation']['p_shear'],
                shear_range=config['augmentation']['shear_range'],
                mode=['bilinear', 'nearest'], padding_mode='zeros'
            ),
            RandFlipd(      # Flip axis 0
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][0]
            ),
            RandFlipd(      # Flip axis 1
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][1]
            ),
            RandFlipd(      # Flip axis 2
                keys=['image', 'label'], prob=config['augmentation']['p_flip'],
                spatial_axis=config['augmentation']['flip_axis'][2]
            ),
            RandSpatialCropd(
                keys=['image', 'label'],
                roi_size=patch_size,
                random_size=False, random_center=True
            ),

            # Intensity transformations
            RandGaussianNoised(
                keys=['image'], prob=config['augmentation']['p_gaussian_noise'], 
                mean=config['augmentation']['gaussian_noise_mean'], std=config['augmentation']['gaussian_noise_std']
            ),
            RandGaussianSmoothd(
                keys=['image'], prob=config['augmentation']['p_gaussian_smooth'],
                sigma_x=config['augmentation']['gaussian_smooth_sigma'], 
                sigma_y=config['augmentation']['gaussian_smooth_sigma'],
                sigma_z=config['augmentation']['gaussian_smooth_sigma'],
            ),
            RandScaleIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_scale'],
                factors=config['augmentation']['intensity_scale_factors']
            ),
            RandShiftIntensityd(
                keys=['image'], prob=config['augmentation']['p_intensity_shift'],
                offsets=config['augmentation']['intensity_shift_offsets']
            ),
            RandAdjustContrastd(
                keys=['image'], prob=config['augmentation']['p_adjust_contrast'],
                gamma=config['augmentation']['adjust_contrast_gamma']
            ),
            ToTensord(
                keys=['image', 'label']
            )
        ]
    elif split == 'val':
        transform = [
            ScaleIntensityRanged(
                keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
                # keys=['image'], a_min=config['foreground_voxel_statistics']['percentile_00_5'], 
                # a_max=config['foreground_voxel_statistics']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(
                keys=['image', 'label'], spatial_size=config['shape_statistics']['median'],
                mode=['area', 'nearest']
            ),
            RandSpatialCropd(
                keys=['image', 'label'], roi_size=patch_size,
                random_size=False, random_center=True
            ),
            ToTensord(
                keys=['image', 'label']
            )
        ]
    return Compose(transform)