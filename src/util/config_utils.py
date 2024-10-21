import yaml

class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None

    def load_config(self):
        if self.config is None:
            with open(self.config_file, 'r') as file:
                self.config = yaml.safe_load(file)
        return self.config