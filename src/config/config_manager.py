"""Manages loading and merging configuration and command-line arguments.

This module defines the ConfigManager class, which loads a YAML configuration file,
overrides settings with command-line arguments, validates the configuration against
the schema, and handles version compatibility and migration.
"""

import argparse
import os

import yaml
from pydantic import ValidationError

from config.config_schema import ConfigSchema
from entities.log_manager import LogManager
from util.errors import ConfigurationError


class ConfigManager:
    """
    Manages loading, merging, and validating configuration and command-line arguments.

    This class loads configuration from a YAML file, applies command-line overrides, validates
    the resulting configuration against a predefined schema, and checks version compatibility.
    """

    def __init__(self, config_file: str, command_line_args: argparse.Namespace):
        """
        Initialize the ConfigManager with the specified configuration file and CLI arguments.

        Args:
            config_file (str): Filename of the configuration file (located under "./config").
            command_line_args (argparse.Namespace): Parsed command-line arguments.
        """
        self.logger = LogManager.get_logger(__name__)
        self.args = command_line_args
        self.config: dict = {}
        self.config_file = os.path.join("./config", config_file)
        self._load_config()
        self._override_with_args()

    def _load_config(self) -> None:
        """
        Load configuration settings from a YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded from file: {self.config_file}")
        else:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")

    def _override_with_args(self) -> None:
        """
        Override configuration values with command-line arguments.

        Iterates over all command-line arguments and updates the configuration dictionary
        if a corresponding key is found in any section.
        """
        if self.args:
            for arg in vars(self.args):
                value = getattr(self.args, arg)
                if value is not None:
                    self._update_or_add_key(arg, value)

    def _update_or_add_key(self, key: str, value: str) -> None:
        """
        Update or add a configuration key with a new value.

        Searches for the key in each section of the configuration and updates it if found.
        Logs a warning if the key does not match any existing configuration key (except for known ignored keys).

        Args:
            key (str): The configuration key to update.
            value (str): The new value to assign to the key.
        """
        for section in self.config:
            if key in self.config[section]:
                self.config[section][key] = value
                self.logger.info(f"Overriding configuration: {key} = {value}")
                break
        else:
            if key not in {"config", "mode", "skip_preprocessing"}:
                self.logger.warning(
                    f"Command-line argument '{key}' does not match any configuration key."
                )

    def get_config(self) -> dict:
        """
        Retrieve the merged configuration dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return self.config

    def validate_config(self) -> None:
        """
        Validate the configuration against the schema.

        Raises:
            ConfigurationError: If configuration validation fails.
        """
        try:
            ConfigSchema(**self.config)
            self.logger.info("Configuration validated successfully.")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation error: {e}") from e

    def is_version_compatible(self) -> None:
        """
        Check if the configuration version is compatible with the current software.

        If the configuration version is older, attempts to migrate settings. If it is newer,
        raises a ConfigurationError.

        Raises:
            ConfigurationError: If the configuration version is unspecified or incompatible.
        """
        meta = self.config.get("meta", {})
        version = meta.get("config_version")
        if not version:
            raise ConfigurationError("Configuration version not specified.")
        self.logger.info(f"Configuration file version: {version}")
        current_version = ConfigSchema().meta.config_version
        if version > current_version:
            raise ConfigurationError(
                f"Configuration file version ({version}) is newer than supported ({current_version}). Please update your software."
            )
        if version < current_version:
            self.logger.warning(
                f"Configuration file version ({version}) is older than expected ({current_version}). Attempting migration."
            )
            self._migrate_config(version)
            self._save_config()

    def _migrate_config(self, old_version: int) -> None:
        """
        Migrate the configuration to the current version.

        Uses a mapping of migration functions to incrementally update the configuration.

        Args:
            old_version (int): The current version of the configuration.

        Raises:
            ConfigurationError: If no migration path exists from the current version.
        """
        migration_steps = {
            1: self._migrate_from_1_to_2,
            # Additional migration functions can be added here.
        }
        current_version = old_version
        while current_version != ConfigSchema().meta.config_version:
            migration_function = migration_steps.get(current_version)
            if migration_function is None:
                raise ConfigurationError(
                    f"No migration path from version {current_version} to {ConfigSchema().meta.config_version}."
                )
            migration_function()
            current_version = self.config["meta"]["config_version"]
            self.logger.info(f"Configuration migrated to version {current_version}.")
        self.logger.info(
            f"Configuration successfully migrated from {old_version} to {current_version}."
        )

    def _migrate_from_1_to_2(self) -> None:
        """
        Migrate the configuration from version 1 to version 2.

        This example migration adds a default seed value to the general section and updates the version.
        """
        self.logger.info("Migrating configuration from version 1 to 2.")
        self.config.setdefault("general", {})
        self.config["general"]["seed"] = 42
        self.config["meta"]["config_version"] = 2

    def _save_config(self) -> None:
        """
        Save the updated configuration back to the YAML file.

        Writes the current configuration dictionary to the original configuration file.
        """
        with open(self.config_file, "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, sort_keys=False)

    def get_config_path(self) -> str:
        """
        Return the path to the configuration file.

        Returns:
            str: The configuration file path.
        """
        return self.config_file
