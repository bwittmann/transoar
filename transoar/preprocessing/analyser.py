"""Module to analyse properties of the dataset."""

import logging

import numpy as np
from tqdm import tqdm 

from transoar.utils.io import load_case


logging.basicConfig(level=logging.INFO)

class DataSetAnalyser:
    """Analyser to analyse properties of dataset."""

    def __init__(self, paths_to_cases):
        self._paths_to_cases = paths_to_cases

        # Init structures to collect properties
        self._shapes = []
        self._spacings = []
        self._foreground_voxels = []


    def analyse(self):
        logging.info('Analyse dataset properties.')
        # Loop over cases and determine properties
        for case in tqdm(self._paths_to_cases):
            loaded_case = load_case(list(case.iterdir()))
            if loaded_case == None:
                continue

            self._shapes.append(loaded_case['data'].shape[1:])
            self._spacings.append(loaded_case['meta_data']['spacing'])

            # Get voxels from foreground
            voxels_foreground = self._get_foreground_voxels(loaded_case['data'])
            self._foreground_voxels += voxels_foreground

        voxel_statistics = self._get_voxel_statistics()

        return None

        
    def _get_foreground_voxels(self, loaded_case_data, subsample=10):
        data, seg = loaded_case_data[0], loaded_case_data[1]
        mask = seg > 0
        return list(data[mask.astype(bool)][::subsample])

    def _get_voxel_statistics(self):
        voxel_statistics = {
            "median": np.median(self._foreground_voxels),
            "mean": np.mean(self._foreground_voxels),
            "std": np.std(self._foreground_voxels),
            "min": np.min(self._foreground_voxels),
            "max": np.max(self._foreground_voxels),
            "percentile_99_5": np.percentile(self._foreground_voxels, 99.5),
            "percentile_00_5": np.percentile(self._foreground_voxels, 00.5),
        }
        return voxel_statistics