"""Script to prepare dataset."""

import logging
import sys

from pathlib import Path
import random

from transoar.utils.io import get_config, set_root_logger
from transoar.data.processor import Preprocessor
from transoar.data.analyzer import DataSetAnalyzer

# TODO: Add multiprocessing

if __name__ == "__main__":
    random.seed(10)  # Set arbitrary seed to make experiments reproducible
    
    # Set config of root logger
    set_root_logger('./logs/prepare_dataset.log')
    logging.info('Started preparing dataset.')
    
    # Load data config
    data_config = get_config('data')

    dataset_name = data_config['dataset_name']
    modality = data_config['modality']
    path_to_gc_dataset = Path(data_config['path_to_gc_dataset'])   # GC dataset for test and val set
    path_to_sc_dataset = Path(data_config['path_to_sc_dataset'])   # SC dataset for train

    path_to_splits = Path(f"./dataset/{dataset_name}_{modality}")

    # Get paths to cases in GC and SC dataset
    cases_sc = list(Path(path_to_sc_dataset).iterdir())
    cases_gc = list(Path(path_to_gc_dataset).iterdir())
    random.shuffle(cases_gc)
    random.shuffle(cases_sc)

    # Create test, val, and train split
    test_set = cases_gc[:int(len(cases_gc)/2)]
    val_set = cases_gc[int(len(cases_gc)/2):]
    train_set = cases_sc

    # Analyze properties of dataset like spacing and intensity properties
    analyzer = DataSetAnalyzer(train_set + val_set, data_config)
    dataset_analysis = analyzer.analyze()

    # Prepare dataset based on dataset analysis
    preprocessor = Preprocessor(
        train_set, val_set, test_set, data_config, path_to_splits, dataset_analysis
    )
    preprocessor.prepare_sets()
