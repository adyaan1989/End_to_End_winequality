import os
from box.exceptions import BoxValueError
import yaml
from winequality.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

def read_yaml(yaml_path: Path) -> ConfigBox:
    try:
        with open(yaml_path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {yaml_path}")
            logger.info(f"YAML file loaded successfully: {yaml_path}")
            return ConfigBox(content)
        
    except FileNotFoundError:
        logger.error(f"YAML file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {yaml_path}")
        raise ValueError(f"Invalid YAML file: {yaml_path}") from e

def create_directories(path_directories: list, verbose=True):
    for path in path_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} KB"

