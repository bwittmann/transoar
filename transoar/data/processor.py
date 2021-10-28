"""Module to pre-process raw cases."""

import logging

import numpy as np
from tqdm import tqdm

from transoar.utils.io import load_case


logging.basicConfig(level=logging.INFO)

class Preprocessor:
    """Preprocessor to pre-process raw data samples."""

    def __init__(
        self, paths_to_train, paths_to_val, paths_to_test, data_config, 
        path_to_splits, analysis
    ):
        self._data_config = data_config
        self._path_to_splits = path_to_splits
        self._analysis = analysis

        self._splits = {
            'train': paths_to_train,
            'val': paths_to_val ,
            'test': paths_to_test
        }

    def prepare_sets(self):
        
        for split_name, split_paths in self._splits.items():
            logging.info(f'Preparing {split_name} set.')
            for case in tqdm(split_paths):
                loaded_case = load_case(list(case.iterdir()))
                
                if loaded_case == None:
                    continue

                # Save actual data and meta data
                # np.savez_compressed(path_to_splits / name / f"{case}.npz", data=loaded_case['data'])
                # with open(path_to_splits / name / f"{case}.pkl", 'wb') as f:
                #     pickle.dump(loaded_case['meta_data'], f)