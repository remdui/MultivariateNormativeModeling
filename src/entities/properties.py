# entities/properties.py

"""The properties module defines the Properties singleton to manage the configuration properties."""

from typing import Any

from pydantic import BaseModel

from config.config_schema import (
    ConfigSchema,
    DatasetConfig,
    GeneralConfig,
    MetaConfig,
    ModelConfig,
    SchedulerConfig,
    SystemConfig,
    TrainConfig,
)
from util.errors import ConfigurationError


class Properties:
    """Singleton class to manage the configuration properties."""

    _instance = None

    # Define the properties for each section. This is necessary for the IDE to provide autocompletion.
    dataset: DatasetConfig
    general: GeneralConfig
    meta: MetaConfig
    model: ModelConfig
    scheduler: SchedulerConfig
    system: SystemConfig
    train: TrainConfig

    @classmethod
    def initialize(cls, config: dict) -> None:
        """Initializes the Properties object only once."""
        if cls._instance is None:
            cls._instance = Properties(config)

    @classmethod
    def get_instance(cls) -> "Properties":
        """Returns the initialized Properties instance."""
        if cls._instance is None:
            raise ConfigurationError("Properties have not been initialized.")
        return cls._instance

    def __init__(self, config: dict) -> None:
        """Initialize the Properties object with the validated ConfigSchema."""
        self._sections: dict[str, BaseModel] = {}

        # Initialize sections and class attributes
        self.__init_section_properties(config)

        # Lock the object to prevent further modification
        self._locked = True

    def __init_section_properties(self, config: dict) -> None:
        """Initialize the section properties with the provided ConfigSchema instance."""
        for section_name, section_class in ConfigSchema.__annotations__.items():
            section_config = config.get(section_name, {})
            section_instance = section_class(**section_config)
            self._sections[section_name] = section_instance
            setattr(self, section_name, section_instance)

    def __setattr__(self, key: str, value: Any) -> None:
        """Prevent modifications to existing attributes after initialization."""
        if hasattr(self, "_locked") and self._locked and hasattr(self, key):
            raise AttributeError(f"Cannot modify immutable property '{key}'")
        super().__setattr__(key, value)

    def __repr__(self) -> str:
        """String representation of all sections for debugging purposes."""
        sections_repr = "\n".join(f"{k}: {v}" for k, v in self._sections.items())
        return f"Properties:\n{sections_repr}"

    @property
    def model_name(self) -> str:
        """Construct the model name from meta information."""
        return f"{self.meta.name}_v{self.meta.version}"
