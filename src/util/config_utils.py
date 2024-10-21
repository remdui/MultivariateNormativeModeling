import os
import yaml

from entities.properties import Properties


class ConfigManager:
    """Manages loading and merging config and arguments, and returns a Properties object."""
    def __init__(self, config_file=None, command_line_args=None):
        self.config = {}
        self.args = command_line_args
        self.load_config(config_file)
        self.override_with_args()

    def load_config(self, config_file):
        """Load the configuration from a YAML file."""

        # config files are located in ./config
        config_file = os.path.join('./config', config_file)

        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file {config_file} not found.")

    def override_with_args(self):
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

    def check_validity(self):
        """Perform validity checks on parameters."""
        if 'action' not in self.config or self.config['action'] not in ['train', 'validate', 'inference']:
            raise ValueError("Invalid or missing 'action' parameter. Must be 'train', 'validate', or 'inference'.")
        if self.config['action'] == 'inference' and not self.config.get('checkpoint'):
            raise ValueError("Checkpoint must be provided for inference.")

    def get_properties(self):
        """Return a Properties object containing the merged config and args."""
        return Properties(self.config)