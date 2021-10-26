"""Script to visualize data of the NIfTI format."""

from pathlib import Path

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd
)

from transoar.utils.visualization import visualize_voxel_grid

NUMBER = '077'
SET = 'SC'
PATH_TO_DATA = Path(f'/home/bastian/Datasets/CT_{SET}/10000{NUMBER}_1/10000{NUMBER}_1_CT_wb.nii.gz')
PATH_TO_SEG = Path(f'/home/bastian/Datasets/CT_{SET}/10000{NUMBER}_1/10000{NUMBER}_1_CT_wb_seg.nii.gz')
# PATH_TO_DATA = Path(f'/home/bastian/Datasets/CT_{SET}/10000{NUMBER}_1/10000{NUMBER}_1_CTce_ThAb.nii.gz')
# PATH_TO_SEG = Path(f'/home/bastian/Datasets/CT_{SET}/10000{NUMBER}_1/10000{NUMBER}_1_CTce_ThAb_seg.nii.gz')

data_dicts = {"image": PATH_TO_DATA, "label": PATH_TO_SEG}

prep_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1, 1, 1), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
)

loaded_image = prep_transforms(data_dicts)
print(loaded_image['image'].shape)
visualize_voxel_grid(loaded_image['image'])
