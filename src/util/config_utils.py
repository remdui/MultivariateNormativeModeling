"""Utility functions for configuration management."""

import yaml

from config.config_schema import (
    DatasetConfig,
    GeneralConfig,
    MetaConfig,
    ModelConfig,
    SchedulerConfig,
    SystemConfig,
    TrainConfig,
)


def extract_config(section_class: type) -> dict:
    """Extract configuration for a given section."""
    instance = section_class()
    return instance.model_dump()


def create_default_config() -> None:
    """Create a default configuration file based on the schema."""
    file_path = "./config/config_default.yml"

    default_config = {
        "system": extract_config(SystemConfig),
        "general": extract_config(GeneralConfig),
        "meta": extract_config(MetaConfig),
        "dataset": extract_config(DatasetConfig),
        "train": extract_config(TrainConfig),
        "model": extract_config(ModelConfig),
        "scheduler": extract_config(SchedulerConfig),
    }

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(default_config, file, sort_keys=False)
