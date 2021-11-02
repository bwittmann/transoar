"""Helper script to prepare GC and SC datasets to nnDetection format."""

import json
import logging
import os
from pathlib import Path
import pickle
import random
import shutil

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from transoar.utils.io import load_nifti, write_nifti


def get_instances(labels_path):
    loaded_labels = load_nifti(labels_path)
    labels = loaded_labels['data']
    label_meta_data = loaded_labels['meta_data']

    classes = np.unique(labels)[1:] - 1 # Because classes should start from 0
    instances = list(range(1, len(classes) + 1))

    instances2classes = {str(idx): int(class_) for idx, class_ in zip(instances, classes)}

    # Change annotation to represent instance rather than a specific class
    for instance, class_ in instances2classes.items():
        labels[labels == (class_ + 1)] = int(instance)

    return {'instances': instances2classes}, labels, label_meta_data


if __name__ == "__main__":
    random.seed(5)  # Set arbitrary seed to make experiments reproducible
    logging.basicConfig(level=logging.INFO)

    PATH_TO_GC_DATASET = Path('/home/bastian/Datasets/CT_GC')   # GC dataset for test and val set
    PATH_TO_SC_DATASET = Path('/home/bastian/Datasets/CT_SC')   # SC dataset for train

    PATH_TO_NEW_DATASET = Path(os.environ['det_data'])
    NEW_NAME = 'Task101_OrganDet'

    DATASET_INFO = {
        "task": NEW_NAME,

        "name": "OrganDet",
        "dim": 3,

        "modalities": {
            "0": "CT"
        },

        "target_class": None,
        "test_labels": True,

        "labels": {
            '0': 'liver',
            '1': 'spleen',
            '2': 'pancreas',
            '3': 'gall_bladder',
            '4': 'urinary_bladder',
            '5': 'aorta',
            '6': 'trachea',
            '7': 'right_lung',
            '8': 'left_lung',
            '9': 'sternum',
            '10': 'thyroid_gland',
            '11': 'first_lumbar_vertebra',
            '12': 'right_kidney',
            '13': 'left_kidney',
            '14': 'right_adrenal_gland',
            '15': 'left_adrenal_gland',
            '16': 'right_psoas_major',
            '17': 'left_psoas_major',
            '18': 'right_rectus_abdominis',
            '19': 'left_rectus_abdominis'
        }
    }

    # Get paths to cases in GC and SC dataset
    cases_sc = list(Path(PATH_TO_SC_DATASET).iterdir())
    cases_gc = list(Path(PATH_TO_GC_DATASET).iterdir())
    random.shuffle(cases_gc)

    # Create test, val, and train split
    test_set = cases_gc[:int(len(cases_gc)/2)]
    val_set = cases_gc[int(len(cases_gc)/2):]
    train_set = cases_sc

    # Combine train and val set since 5 fold cross-validation
    train_set += val_set

    # Create target dirs
    target_dir_tr_data = PATH_TO_NEW_DATASET / NEW_NAME / 'raw_splitted' / 'imagesTr'
    target_dir_tr_labels = PATH_TO_NEW_DATASET / NEW_NAME / 'raw_splitted' / 'labelsTr'
    target_dir_ts_data = PATH_TO_NEW_DATASET / NEW_NAME / 'raw_splitted' / 'imagesTs'
    target_dir_ts_labels = PATH_TO_NEW_DATASET / NEW_NAME / 'raw_splitted' / 'labelsTs'

    targe_dirs = [target_dir_tr_data, target_dir_tr_labels, target_dir_ts_data, target_dir_ts_labels]
    for target_dir in targe_dirs:
        os.makedirs(target_dir)

    # random.shuffle(train_set)
    # random.shuffle(test_set)
    # train_set = train_set[:20]
    # test_set = test_set[:5]

    logging.info('Preparing dataset to match nndet format.')
    for idx, case in enumerate(tqdm((train_set + test_set))):
        case_paths = list(case.iterdir())
        case_paths.sort(key=lambda x: len(str(x)))
        data_path, labels_path = case_paths

        if idx < len(train_set):
            target_dir_data = target_dir_tr_data
            target_dir_labels = target_dir_tr_labels
        else:
            target_dir_data = target_dir_ts_data
            target_dir_labels = target_dir_ts_labels

        # Copy labels and generate instances json
        try:
            instances, labels, labels_meta_data = get_instances(labels_path)
        except RuntimeError:
            continue

        # Write labels
        write_nifti(labels, labels_meta_data, target_dir_labels / ('case_' + f'{idx:03}' + '.nii.gz'))
        with open(target_dir_labels / ('case_' + f'{idx:03}' + '.json'), 'w') as outfile:
            json.dump(instances, outfile, indent=3)
        
        # Write data
        shutil.copy(data_path, target_dir_data / ('case_' + f'{idx:03}' + '_0000.nii.gz'))

    # Dump dataset info json
    with open(PATH_TO_NEW_DATASET / NEW_NAME / 'dataset.json', 'w') as outfile:
        json.dump(DATASET_INFO, outfile, indent=3)
