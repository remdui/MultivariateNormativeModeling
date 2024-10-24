import yaml

from config.config_schema import ConfigSchema


def create_default_config():
    """Create a default configuration file based on the schema."""
    file_path = "./config/config_default.yml"

    default_config = {
        "system": {
            key: getattr(ConfigSchema.System, key)
            for key in dir(ConfigSchema.System)
            if not key.startswith("__")
        },
        "general": {
            key: getattr(ConfigSchema.General, key)
            for key in dir(ConfigSchema.General)
            if not key.startswith("__")
        },
        "meta": {
            key: getattr(ConfigSchema.Meta, key)
            for key in dir(ConfigSchema.Meta)
            if not key.startswith("__")
        },
        "dataset": {
            key: getattr(ConfigSchema.Dataset, key)
            for key in dir(ConfigSchema.Dataset)
            if not key.startswith("__")
        },
        "train": {
            key: getattr(ConfigSchema.Train, key)
            for key in dir(ConfigSchema.Train)
            if not key.startswith("__")
        },
        "model": {
            key: getattr(ConfigSchema.Model, key)
            for key in dir(ConfigSchema.Model)
            if not key.startswith("__")
        },
        "scheduler": {
            key: getattr(ConfigSchema.Scheduler, key)
            for key in dir(ConfigSchema.Scheduler)
            if not key.startswith("__")
        },
    }
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(default_config, file)
