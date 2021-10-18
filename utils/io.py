"""Helper functions for input/output."""

from pathlib import Path

import yaml

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



if __name__ == "__main__":
    pass
