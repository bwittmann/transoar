"""Script to prepare dataset."""

import logging
import os
from pathlib import Path
import pickle
import random

import numpy as np
from tqdm import tqdm

from transoar.utils.io import load_case
from transoar.preprocessing.processor import Preprocessor
from transoar.preprocessing.analyser import DataSetAnalyser

# TODO: Add multiprocessing

def prepare_set(name, path_to_dataset, path_to_splits, cases):
    logging.info(f'Preparing {name} set.')
    for case in tqdm(cases):
        loaded_case = load_case([path.absolute() for path in Path(path_to_dataset / case).glob('**/*')])
        
        if loaded_case == None:
            continue

        # Save actual data and meta data
        # np.savez_compressed(path_to_splits / name / f"{case}.npz", data=loaded_case['data'])
        # with open(path_to_splits / name / f"{case}.pkl", 'wb') as f:
        #     pickle.dump(loaded_case['meta_data'], f)

if __name__ == "__main__":
    random.seed(5)  # Set arbitrary seed to make experiments reproducible
    logging.basicConfig(level=logging.INFO)

    DATASET_NAME = 'transoar'
    MODALITY = 'CT'
    PATH_TO_GC_DATASET = Path('/home/bastian/Datasets/CT_GC')   # GC dataset for test and val set
    PATH_TO_SC_DATASET = Path('/home/bastian/Datasets/CT_SC')   # SC dataset for train

    path_to_splits = Path(f"./data/{DATASET_NAME}_{MODALITY}")

    # Get paths to cases in GC and SC dataset
    cases_sc = [path for path in Path(PATH_TO_SC_DATASET).iterdir()]
    cases_gc = [path for path in Path(PATH_TO_GC_DATASET).iterdir()]
    random.shuffle(cases_gc)

    # Create test, val, and train split
    test_set = cases_gc[:int(len(cases_gc)/2)]
    val_set = cases_gc[int(len(cases_gc)/2):]
    train_set = cases_sc

    # Analyse properties of dataset like spacing and intensity properties
    analyser = DataSetAnalyser(paths_to_cases=train_set + val_set)
    dataset_analysis = analyser.analyse()






    # Prepare SC dataset (train) and GC dataset (test and val)
    dataset_infos = zip(
        ['train', 'val', 'test'],
        [PATH_TO_SC_DATASET, PATH_TO_GC_DATASET, PATH_TO_GC_DATASET],
        [path_to_splits, path_to_splits, path_to_splits],
        [train_set, val_set, test_set]
    )
    
    for dataset_info in dataset_infos:
        os.makedirs(path_to_splits / dataset_info[0])
        prepare_set(*dataset_info)
