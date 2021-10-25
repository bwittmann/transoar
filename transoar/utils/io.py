"""Helper functions for input/output."""

import logging
from pathlib import Path

import numpy as np
import yaml
import SimpleITK as sitk

PATH_TO_CONFIG = Path("./config/")
logging.basicConfig(level=logging.INFO)

def get_complete_config():
    """Loads .yaml files specified in ./config/main.yaml.

    Returns:
        A dict containing the parameters specified in the included individual
        config files.
    """
    config = {}

    # Load includes
    main = get_config('main')

    # Add includes
    for config_file in main:
        config_to_include = get_config(config_file)
        config[config_file] = config_to_include

    return config

def get_config(config_name):
    """Loads a .yaml file from ./config corresponding to the name arg.

    Args:
        config_name: A string referring to the .yaml file to load.

    Returns:
        A container including the information of the referred .yaml file.
    """
    with open(PATH_TO_CONFIG / (config_name + '.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def load_nifti(path_to_file):
    """Reads a .nii.gz file and extracts relevant informations.
    
    Returns:
        A dict containing the data in form of a np.ndarray and the meta data."""
    data_itk = sitk.ReadImage(str(path_to_file))

    # Extract relevant information
    data = sitk.GetArrayFromImage(data_itk).astype(np.float32)
    
    meta_data = {
        'original_size_of_data': data.shape,
        'original_spacing': np.array(data_itk.GetSpacing())[[2, 1, 0]],
        'itk_origin': data_itk.GetOrigin(),
        'itk_spacing': data_itk.GetSpacing(),
        'itk_direction': data_itk.GetDirection(),
    }

    return {'data': data, 'meta_data': meta_data}

def write_nifti(data, meta_data, file_path):
    data_itk = sitk.GetImageFromArray(data)
    data_itk.SetOrigin(meta_data['itk_origin'])
    data_itk.SetSpacing(meta_data['itk_spacing'])
    data_itk.SetDirection(meta_data['itk_direction'])

    sitk.WriteImage(data_itk, str(file_path))

def load_case(case_paths):
    """ Loads relevant data from a complete case consisting of the 
    data and the segmentation labels.
    
    Args:
        case_paths: A list containing the paths to the data and labels.

    Returns:
        pass
        """
    # Sort paths for labels to be the second path
    case_paths.sort(key=lambda x: len(str(x)))

    data = []
    try:
        for path in case_paths:
            loaded_case = load_nifti(path)
            data.append(loaded_case['data'])
    except RuntimeError:
        logging.warning(f'Skipped case {path}.')
        return None

    # Cat data and labels to store efficiently
    loaded_case['data'] = np.stack(data)

    # Add label -1 for voxels that do not store data
    # loaded_case['data'][1][(loaded_case['data'][1] == 0.) & (loaded_case['data'][0] == 0.)] = -1.

    # Add additional information to meta data
    loaded_case['meta_data']['classes'] = [int(class_) for class_ in np.unique(loaded_case['data'][1])][1:]
    num_classes = len(loaded_case['meta_data']['classes'])

    # Maps the individual instances to the corresponding class
    loaded_case['meta_data']['instances'] = {
        str(key): int(value) for key, value in zip(range(1, num_classes+1), loaded_case['meta_data']['classes'])
    }

    return loaded_case

def write_pkl():
    pass
