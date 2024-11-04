"""Utility functions for configuration management."""

import os

import yaml

from config.config_schema import (
    DatasetConfig,
    GeneralConfig,
    MetaConfig,
    ModelConfig,
    SystemConfig,
    TrainConfig,
)
from entities.log_manager import LogManager


def extract_config(section_class: type) -> dict:
    """Extract configuration for a given section."""
    instance = section_class()
    return instance.model_dump()


def create_default_config() -> None:
    """Create a default configuration file based on the schema."""

    logger = LogManager.get_logger(__name__)

    # If config directory does not exist, create it
    if not os.path.exists("./config"):
        os.makedirs("./config")
        logger.info("Created config directory")

    # Define the default configuration file path
    file_path = "./config/config_default.yml"

    # Extract configuration for each section from the schema
    default_config = {
        "system": extract_config(SystemConfig),
        "general": extract_config(GeneralConfig),
        "meta": extract_config(MetaConfig),
        "dataset": extract_config(DatasetConfig),
        "train": extract_config(TrainConfig),
        "model": extract_config(ModelConfig),
    }

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(default_config, file, sort_keys=False)
        logger.info(f"Generated default configuration file from schema: {file_path}")
