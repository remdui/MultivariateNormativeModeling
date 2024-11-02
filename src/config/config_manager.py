# config/config_manager.py

"""Manages loading and merging config and arguments."""

import argparse
import os

import yaml
from pydantic import ValidationError

from config.config_schema import ConfigSchema, MetaConfig
from util.errors import ConfigurationError
from util.log_utils import log_message


class ConfigManager:
    """Manages loading and merging config and arguments."""

    def __init__(self, config_file: str, command_line_args: argparse.Namespace):
        """Initialize the ConfigManager with the provided configuration file and command-line arguments."""
        self.config_file = config_file
        self.config: dict = {}
        self.args = command_line_args
        self._load_config()
        self._override_with_args()
        self.validate_config()

    def _load_config(self) -> None:
        """Load the configuration from a YAML file."""
        config_file = os.path.join("./config", self.config_file)

        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Configuration file {config_file} not found.")

    def _override_with_args(self) -> None:
        """Override configuration values with command-line arguments."""
        if self.args:
            for arg in vars(self.args):
                value = getattr(self.args, arg)
                if value is not None:
                    # Update the config with the value from command-line arguments
                    self._update_or_add_key(arg, value)

    def _update_or_add_key(self, key: str, value: str) -> None:
        """Update or add a key in the nested configuration dictionary."""
        for section in self.config:
            if key in self.config[section]:
                self.config[section][key] = value
                break
        else:
            # Key not found in any section
            if key not in {"config", "mode"}:
                log_message(
                    f"Warning: Command-line argument '{key}' does not match any configuration key."
                )

    def get_config(self) -> dict:
        """Return the validated configuration as a ConfigSchema instance."""
        return self.config

    def validate_config(self) -> None:
        """Validate the configuration dictionary and store the ConfigSchema instance."""
        try:
            ConfigSchema(**self.config)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation error: {e}") from e

    def is_version_compatible(self) -> None:
        """Check if the configuration file is compatible with the current software version.

        Migrate the configuration if necessary.
        """
        meta = self.config.get("meta", {})
        version = meta.get("config_version")
        if not version:
            raise ConfigurationError("Version not specified in the configuration file.")
        if version > MetaConfig.config_version:
            raise ConfigurationError(
                f"Configuration file version ({version}) is newer than supported ({MetaConfig.config_version}). Please update your software."
            )
        if version < MetaConfig.config_version:
            log_message(
                f"Configuration file version ({version}) is older than schema ({MetaConfig.config_version}). Attempting to migrate settings."
            )
            self._migrate_config(version)
            self._save_config()

    def _migrate_config(self, old_version: int) -> None:
        """Migrate the configuration to the current version."""
        migration_steps = {
            1: self._migrate_from_1_to_2,
            # Add more mappings as needed
        }
        current_version = old_version

        while current_version != MetaConfig.config_version:
            migration_function = migration_steps.get(current_version)
            if not migration_function:
                raise ConfigurationError(
                    f"No migration path from version {current_version} to {MetaConfig.config_version}."
                )
            migration_function()
            current_version = self.config["meta"]["config_version"]
            log_message(f"Configuration migrated to version {current_version}.")
        log_message(
            f"Configuration successfully migrated from {old_version} to version {current_version}."
        )

    def _migrate_from_1_to_2(self) -> None:
        """Migrate the configuration from version 1 to version 2."""
        log_message("Migrating configuration from version 1 to 2")

        # Add missing sections or keys
        self.config.setdefault("general", {})
        self.config["general"]["seed"] = 42

        # Update the configuration version
        self.config["meta"]["config_version"] = 2

    def _save_config(self) -> None:
        """Save the updated configuration to the file."""
        config_file = os.path.join("./config", self.config_file)
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, sort_keys=False)
