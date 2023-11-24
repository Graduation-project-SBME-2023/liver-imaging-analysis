"""module for reading configuration from the json file."""

import json
from logger import setup_logger
import logging
setup_logger()
logger = logging.getLogger(__name__)

class Config:
    """class for reading configuration from the json file."""


    def __init__(self) -> None:
        """Init for the class."""

        logger.info("Loading configuration from configs.json")

        with open(
            "liver_imaging_analysis/config/configs.json"
        ) as json_path:
            config_dict = json.load(json_path)

        for key in config_dict:

            logger.debug("Setting config param '%s' to %s", key, config_dict[key])
            setattr(self, key, config_dict[key])

        logger.info("Configuration loaded")

config = Config()
