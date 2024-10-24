import os

import yaml


class ConfigManager:
    """Manages loading and merging config and arguments, and returns a Properties object."""

    def __init__(self, config_file=None, command_line_args=None):
        self.config = {}
        self.args = command_line_args
        self._load_config(config_file)
        self._override_with_args()

    def _load_config(self, config_file):
        """Load the configuration from a YAML file."""

        # config files are located in ./config
        config_file = os.path.join("./config", config_file)

        if config_file and os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file {config_file} not found.")

    def _override_with_args(self):
        """Override configuration values with command-line arguments."""
        if self.args:
            for arg in vars(self.args):
                value = getattr(self.args, arg)
                if value is not None:
                    # Update the config with the value from command-line arguments
                    self._update_or_add_key(arg, value)

    def _update_or_add_key(self, key, value):
        """Update or add a key in the nested configuration dictionary."""
        found = False
        for section in self.config:
            if key in self.config[section]:
                self.config[section][key] = value
                found = True
                break
        if not found:
            self.config[key] = value

    def get_config(self):
        """Return a Properties object containing the merged config and args."""
        return self.config
