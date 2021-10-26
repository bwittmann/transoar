"""Script to filter out problematic cases."""

from pathlib import Path

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,

)

from transoar.utils.io import get_config 

data_config = get_config('data')
PATH_TO_GC_DATASET = Path(data_config['path_to_gc_dataset'])
PATH_TO_SC_DATASET = Path(data_config['path_to_sc_dataset'])
cases_sc = list(Path(PATH_TO_SC_DATASET).iterdir())
cases_gc = list(Path(PATH_TO_GC_DATASET).iterdir())

cases_wrong_dim = []
cases_twisted = []
for case in cases_gc + cases_sc:
    PATH_TO_DATA, PATH_TO_SEG = list(case.iterdir())

    data_dicts = {"image": PATH_TO_DATA, "label": PATH_TO_SEG}

    prep_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
    )

    prep_transforms_w = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    loaded_image = prep_transforms(data_dicts)
    shape1 = loaded_image['image'].shape

    loaded_image = prep_transforms_w(data_dicts)
    shape2 = loaded_image['image'].shape

    if shape1[1] != shape1[2]:
        cases_twisted.append(PATH_TO_DATA)

    if shape1 != shape2:
        cases_wrong_dim.append(PATH_TO_DATA)
    print(shape1, shape2)
    print(15*"-")
print(cases_wrong_dim)
print(15*"-")
print(cases_twisted)