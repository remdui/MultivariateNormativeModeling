"""Utility functions for configuration management."""

import os

import yaml

from config.config_schema import (
    DataAnalysisConfig,
    DatasetConfig,
    GeneralConfig,
    MetaConfig,
    ModelConfig,
    SystemConfig,
    TrainConfig,
    ValidationConfig,
)
from entities.log_manager import LogManager

# Constants for configuration directory and default file path.
CONFIG_DIR = os.path.join(".", "config")
CONFIG_DEFAULT_FILE = os.path.join(CONFIG_DIR, "config_default.yml")


def extract_config(section_class: type) -> dict:
    """
    Extract configuration from a given schema section.

    This function instantiates the provided configuration schema class
    and returns its dictionary representation using the `model_dump()` method.

    Args:
        section_class (Type): A configuration schema class with a `model_dump()` method.

    Returns:
        dict: The configuration data as a dictionary.
    """
    instance = section_class()
    return instance.model_dump()


def create_default_config() -> None:
    """
    Generate a default configuration YAML file based on the defined schema.

    Aggregates configuration from various schema sections (system, general,
    meta, dataset, data_analysis, train, model, and validation) and writes the
    combined configuration to a default YAML file located at:
        './config/config_default.yml'

    The configuration directory is created if it does not already exist.

    Raises:
        OSError: If there is an issue creating the directory or writing the file.
    """
    logger = LogManager.get_logger(__name__)

    # Ensure the configuration directory exists.
    os.makedirs(CONFIG_DIR, exist_ok=True)
    logger.info("Ensured config directory exists")

    # Aggregate configuration sections.
    default_config = {
        "system": extract_config(SystemConfig),
        "general": extract_config(GeneralConfig),
        "meta": extract_config(MetaConfig),
        "dataset": extract_config(DatasetConfig),
        "data_analysis": extract_config(DataAnalysisConfig),
        "train": extract_config(TrainConfig),
        "model": extract_config(ModelConfig),
        "validation": extract_config(ValidationConfig),
    }

    # Write the default configuration to a YAML file using safe_dump.
    with open(CONFIG_DEFAULT_FILE, "w", encoding="utf-8") as file:
        yaml.safe_dump(default_config, file, sort_keys=False)

    logger.info(
        f"Generated default configuration file from schema: {CONFIG_DEFAULT_FILE}"
    )
