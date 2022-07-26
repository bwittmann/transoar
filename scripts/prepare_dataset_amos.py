"""Script to prepare amos dataset."""

import logging

from pathlib import Path
import random

from transoar.utils.io import get_config, set_root_logger, load_json
from transoar.data.preprocessor_amos import PreProcessor


if __name__ == "__main__":
    # Set config of root logger
    set_root_logger('./logs/prepare_dataset.log')
    logging.info('Started preparing dataset.')
    
    # Load data config
    preprocessing_config = get_config('preprocessing_amos')
    data_config = get_config(preprocessing_config['dataset_config'])

    random.seed(preprocessing_config['seed'])  # Set arbitrary seed to make experiments reproducible

    dataset_name = preprocessing_config['dataset_name']
    modality = preprocessing_config['modality']
    path_dataset = Path(preprocessing_config['path_to_dataset'])   # complete dataset 
    path_to_splits = Path(f"./dataset/{dataset_name}_{modality}")

    data_info = load_json(path_dataset / 'task1_dataset.json')

    # Get paths to train cases
    cases = data_info['training']
    random.shuffle(cases)

    # Create test, val, and train split
    train_set = cases[:preprocessing_config['train']]
    val_set = cases[preprocessing_config['train'] : preprocessing_config['train'] + preprocessing_config['val']]
    test_set = cases[preprocessing_config['train'] + preprocessing_config['val']:]

    logging.info(f'Preparing dataset {dataset_name}_{modality}.')
    logging.info(f'len train: {len(train_set)}, len val: {len(val_set)}, len test: {len(test_set)}.')

    # Prepare dataset based on dataset analysis
    logging.info(f"Starting dataset preprocessing. Target shape: {preprocessing_config['resize_shape']}.")
    preprocessor = PreProcessor(
        train_set, val_set, test_set, path_dataset, path_to_splits, preprocessing_config, data_config
    )
    preprocessor.run()
    logging.info(f'Succesfully finished dataset preprocessing.')
