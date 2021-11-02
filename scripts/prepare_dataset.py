"""Script to prepare dataset."""

import logging
from pathlib import Path
import random

from transoar.utils.io import get_config
from transoar.data.processor import Preprocessor
from transoar.data.analyzer import DataSetAnalyzer

# TODO: Add multiprocessing

if __name__ == "__main__":
    random.seed(5)  # Set arbitrary seed to make experiments reproducible
    logging.basicConfig(level=logging.INFO)

    # Load data config
    data_config = get_config('data')

    DATASET_NAME = data_config['dataset_name']
    MODALITY = data_config['modality']
    PATH_TO_GC_DATASET = Path(data_config['path_to_gc_dataset'])   # GC dataset for test and val set
    PATH_TO_SC_DATASET = Path(data_config['path_to_sc_dataset'])   # SC dataset for train

    path_to_splits = Path(f"./dataset/{DATASET_NAME}_{MODALITY}")

    # Get paths to cases in GC and SC dataset
    cases_sc = list(Path(PATH_TO_SC_DATASET).iterdir())
    cases_gc = list(Path(PATH_TO_GC_DATASET).iterdir())
    random.shuffle(cases_gc)

    # Create test, val, and train split
    test_set = cases_gc[:int(len(cases_gc)/2)][:10]
    val_set = cases_gc[int(len(cases_gc)/2):][:10]
    train_set = cases_sc[:10]

    # Analyze properties of dataset like spacing and intensity properties
    analyzer = DataSetAnalyzer(train_set + val_set, data_config)
    dataset_analysis = analyzer.analyze()

    # Prepare dataset based o dataset analysis
    preprocessor = Preprocessor(
        train_set, val_set, test_set, data_config, path_to_splits, dataset_analysis
    )
    preprocessor.prepare_sets()
