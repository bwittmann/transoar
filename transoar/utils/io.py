"""Helper functions for input/output."""

from pathlib import Path

import yaml
import nibabel as nib

PATH_TO_CONFIG = Path("./config/")

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
        A dict with the keys data, meta_data, and affine."""
    nifti_img = nib.load(path_to_file)

    # Extract relevant information
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    meta_data = dict(nifti_img.header)

    return {'data': data, 'meta_data': meta_data, 'affine': affine}


if __name__ == "__main__":
    pass