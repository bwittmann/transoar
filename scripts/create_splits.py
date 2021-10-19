"""Script to create splits."""

import logging
import os
from pathlib import Path
from random import shuffle
import shutil

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

DATASET_NAME = 'transoar'
MODALITY = 'CT'
PATH_TO_GC_DATASET = Path('/home/bastian/Datasets/CT_GC')   # GC dataset for test and val set
PATH_TO_SC_DATASET = Path('/home/bastian/Datasets/CT_SC')   # SC dataset for train

path_to_splits = Path(f"./data/{DATASET_NAME}_{MODALITY}")

# Get names of data samples of GC set and shuffle them
names_sc = os.listdir(PATH_TO_SC_DATASET)
names_gc = os.listdir(PATH_TO_GC_DATASET)
shuffle(names_gc)

# Split shuffled GC set into test and val set
test_set = names_gc[:int(len(names_gc)/2)]
val_set = names_gc[int(len(names_gc)/2):]

# Copy SC dataset to train
logging.info(f'Copying train set to data.')
for item in tqdm(names_sc):
    shutil.copytree(PATH_TO_SC_DATASET / item, path_to_splits / 'train' / item)

# Copy GC dataset to test and val
logging.info(f'Copying val set to data.')
for item in tqdm(val_set):
    shutil.copytree(PATH_TO_GC_DATASET / item, path_to_splits / 'val' / item)

logging.info(f'Copying test set to data.')
for item in tqdm(test_set):
    shutil.copytree(PATH_TO_GC_DATASET / item, path_to_splits / 'test' / item)
