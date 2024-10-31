"""Utility functions for configuration management."""

import yaml

from config.config_schema import ConfigSchema


def extract_config(section: type) -> dict:
    """Extract configuration for a given section."""
    return {
        key: getattr(section, key) for key in dir(section) if not key.startswith("__")
    }


def create_default_config() -> None:
    """Create a default configuration file based on the schema."""
    file_path = "./config/config_default.yml"

    default_config = {
        "system": extract_config(ConfigSchema.System),
        "general": extract_config(ConfigSchema.General),
        "meta": extract_config(ConfigSchema.Meta),
        "dataset": extract_config(ConfigSchema.Dataset),
        "train": extract_config(ConfigSchema.Train),
        "model": extract_config(ConfigSchema.Model),
        "scheduler": extract_config(ConfigSchema.Scheduler),
    }

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(default_config, file)
