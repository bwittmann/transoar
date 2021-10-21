"""Helper functions for input/output."""

import logging
from pathlib import Path

import numpy as np
import yaml
import SimpleITK as sitk

PATH_TO_CONFIG = Path("./config/")
logging.basicConfig(level=logging.INFO)

def get_config():
    """Loads .yaml files specified in ./config/main.yaml.

    Returns:
        A dict containing the parameters specified in the included individual
        config files.
    """
    config = {}

    # Load includes
    with open(PATH_TO_CONFIG / 'main.yaml', 'r') as stream:
        main = yaml.safe_load(stream)

    # Add includes
    for config_file in main['include']:
        with open(PATH_TO_CONFIG / config_file, 'r') as stream:
            config_to_include = yaml.safe_load(stream)

        config[config_file[:config_file.index('.')]] = config_to_include

    return config

def load_nifti(path_to_file):
    """Reads a .nii.gz file and extracts relevant informations.
    
    Returns:
        A dict containing the data in form of a np.ndarray and the meta data."""
    data_itk = sitk.ReadImage(str(path_to_file))

    # Extract relevant information
    data = sitk.GetArrayFromImage(data_itk).astype(np.float32)
    
    meta_data = {
        'origin': data_itk.GetOrigin(), # TODO: swaped dims?
        'spacing': data_itk.GetSpacing(),
        'direction': data_itk.GetDirection(),
    }

    return {'data': data, 'meta_data': meta_data}


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
    loaded_case['data'][1][(loaded_case['data'][1] == 0.) & (loaded_case['data'][0] == 0.)] = -1.

    # Add additional information to meta data
    loaded_case['meta_data']['classes'] = np.unique(loaded_case['data'][1])
    num_classes = len(loaded_case['meta_data']['classes'])

    if -1 in loaded_case['meta_data']['classes']:   # Remove -1, since it is not an instance
        classes = loaded_case['meta_data']['classes'][2:]
    else:
        classes = loaded_case['meta_data']['classes'][1:]

    loaded_case['meta_data']['instances'] = {
        str(key): int(value) for key, value in zip(range(1, num_classes+1), classes)
    }

    return loaded_case


if __name__ == "__main__":
    pass  