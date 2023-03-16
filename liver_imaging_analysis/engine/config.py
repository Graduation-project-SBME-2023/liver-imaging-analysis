"""module for reading configuration from the json file."""

import json


class Config:
    """class for reading configuration from the json file."""

    def __init__(self) -> None:
        """Init for the class."""
        with open(
            "config/configs.json"
        ) as json_path:
            config_dict = json.load(json_path)

        for key in config_dict:
            setattr(self, key, config_dict[key])


config = Config()
