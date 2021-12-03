"""Helper functions for input/output."""

import os
import json
import logging
import pickle
from pathlib import Path
import sys
import socket
import subprocess

import numpy as np
import torch
import yaml
import SimpleITK as sitk

PATH_TO_CONFIG = Path("./config/")


def get_config(config_name):
    """Loads a .yaml file from ./config corresponding to the name arg.

    Args:
        config_name: A string referring to the .yaml file to load.

    Returns:
        A container including the information of the referred .yaml file and information
        regarding the dataset, if specified in the referred .yaml file.
    """
    with open(PATH_TO_CONFIG / (config_name + '.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)

    # Add dataset config
    if 'dataset' in config:
        path_to_data_info = Path(os.getcwd()) / 'dataset' / config['dataset'] / 'data_info.json'
        config.update(load_json(path_to_data_info))

    return config

def load_nifti(path_to_file):
    """Reads a .nii.gz file and extracts relevant information.
    
    Returns:
        A dict containing the data in form of a np.ndarray and the meta data.
    """
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
    # data_itk.SetOrigin(meta_data['itk_origin'])
    data_itk.SetSpacing(meta_data['itk_spacing'])
    # data_itk.SetDirection(meta_data['itk_direction'])

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

def write_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)

def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def set_root_logger(file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s : %(levelname)s [%(module)s, %(lineno)d] %(message)s",
        handlers=[
            logging.FileHandler(file_path, 'w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def get_meta_data():
    meta_data = {}
    meta_data['git_commit_hash'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    meta_data['python_version'] = sys.version.splitlines()[0]
    meta_data['gcc_version'] = sys.version.splitlines()[1]
    meta_data['pytorch_version'] = torch.__version__
    meta_data['host_name'] = socket.gethostname()

    return meta_data
