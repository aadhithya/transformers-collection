import os

import yaml
from loguru import logger
from munch import DefaultMunch


def load_config_file(cfg_path: str):
    assert os.path.exists(cfg_path), f"Config file not found: {cfg_path}"
    try:
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
    except Exception:
        logger.critical(f"Error loading config file: {cfg_path}! Exiting.")
    return DefaultMunch.fromDict(cfg)
