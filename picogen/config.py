from collections import UserDict
from typing import DefaultDict
import yaml

class YamlConfig(UserDict):
    def __init__(self, config_file):
        self.data = DefaultDict(dict)

        config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        for key in config:
            for key2 in config[key]:
                self.data[key][key2] = config[key][key2]
