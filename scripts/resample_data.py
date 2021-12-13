"""Helper script to resample ."""

from pathlib import Path

from tqdm import tqdm
from monai.transforms import (
    Spacingd, 
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SaveImaged
)

from transoar.utils.visualization import visualize_voxel_grid

def resample_nifti(case_path, target_spacing, new_path, debug=True):
    assert isinstance(target_spacing, list)

    # Get path of label and data
    paths = sorted(list(Path(case_path).iterdir()))

    # Init transform
    transform_and_save = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")
        ),
        SaveImaged(
            keys=["image", "label"], resample=False, output_dir=new_path, squeeze_end_dims=True,
            output_postfix="", separate_folder=False

        )
    ])

    # Generate dict required for monai transforms
    data_dict = {
        'image': paths[0],
        'label': paths[1]
    }

    out = transform_and_save(data_dict)

    if debug:
        visualize_voxel_grid(out['image'])
        visualize_voxel_grid(out['label'])


if __name__ == "__main__":
    TARGET_SPACING = [4, 4, 4]
    PATH_TO_NEW_DATASET = Path(f'/home/bastian/datasets/CT_GC_{TARGET_SPACING[0]}mm')
    PATH_TO_DATASET = Path('/home/bastian/datasets/CT_GC')

    cases_path = list(Path(PATH_TO_DATASET).iterdir())

    # Resample
    for case_path in tqdm(cases_path):
        new_path = PATH_TO_NEW_DATASET / case_path.name
        resample_nifti(case_path, TARGET_SPACING, new_path, debug=False)
