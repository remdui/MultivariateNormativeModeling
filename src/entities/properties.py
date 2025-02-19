"""Singleton class to manage configuration properties.

This module defines the Properties singleton that loads and stores configuration settings
from a validated configuration dictionary. The configuration is partitioned into multiple sections
(e.g., dataset, model, system) based on the ConfigSchema.
"""

from typing import Any

from pydantic import BaseModel

from config.config_schema import (
    ConfigSchema,
    DataAnalysisConfig,
    DatasetConfig,
    GeneralConfig,
    MetaConfig,
    ModelConfig,
    SystemConfig,
    TrainConfig,
    ValidationConfig,
)
from util.errors import ConfigurationError


class Properties:
    """
    Singleton class for configuration properties.

    This class loads configuration sections from a dictionary (validated against ConfigSchema)
    and exposes them as attributes. Once initialized, the Properties object becomes immutable.
    """

    _instance = None

    # Section attributes for IDE autocompletion.
    data_analysis: DataAnalysisConfig
    dataset: DatasetConfig
    general: GeneralConfig
    meta: MetaConfig
    model: ModelConfig
    system: SystemConfig
    train: TrainConfig
    validation: ValidationConfig

    @classmethod
    def initialize(cls, config: dict) -> None:
        """
        Initialize the Properties singleton with the given configuration.

        This method should be called once to set up the configuration.

        Args:
            config (dict): The configuration dictionary validated by ConfigSchema.
        """
        if cls._instance is None:
            cls._instance = Properties(config)

    @classmethod
    def get_instance(cls) -> "Properties":
        """
        Get the singleton Properties instance.

        Returns:
            Properties: The initialized Properties instance.

        Raises:
            ConfigurationError: If the Properties have not been initialized.
        """
        if cls._instance is None:
            raise ConfigurationError("Properties have not been initialized.")
        return cls._instance

    @classmethod
    def overwrite_instance(cls, updated_properties: "Properties") -> "Properties":
        """
        Overwrite the existing Properties instance with an updated instance.

        Args:
            updated_properties (Properties): The updated Properties instance.

        Returns:
            Properties: The new Properties instance.

        Raises:
            ConfigurationError: If the Properties have not been initialized.
        """
        if cls._instance is None:
            raise ConfigurationError("Properties have not been initialized.")
        cls._instance = updated_properties
        return cls._instance

    def __init__(self, config: dict) -> None:
        """
        Initialize the Properties instance with the validated configuration.

        Args:
            config (dict): The configuration dictionary.
        """
        self._sections: dict[str, BaseModel] = {}
        self.__init_section_properties(config)
        # Lock the instance to prevent further modification.
        self._locked = True

    def __init_section_properties(self, config: dict) -> None:
        """
        Initialize configuration sections from the provided config dictionary.

        For each section defined in ConfigSchema, instantiate the corresponding configuration
        class with the provided section configuration.

        Args:
            config (dict): The configuration dictionary.
        """
        for section_name, section_class in ConfigSchema.__annotations__.items():
            section_config = config.get(section_name, {})
            section_instance = section_class(**section_config)
            self._sections[section_name] = section_instance
            setattr(self, section_name, section_instance)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Prevent modification of immutable properties after initialization.

        Args:
            key (str): The attribute name.
            value (Any): The new value.

        Raises:
            AttributeError: If attempting to modify an existing property after the instance is locked.
        """
        if hasattr(self, "_locked") and self._locked and hasattr(self, key):
            raise AttributeError(f"Cannot modify immutable property '{key}'")
        super().__setattr__(key, value)

    def __repr__(self) -> str:
        """
        Return a string representation of all configuration sections for debugging.

        Returns:
            str: A formatted string listing each configuration section and its values.
        """
        sections_repr = "\n".join(f"{k}: {v}" for k, v in self._sections.items())
        return f"Properties:\n{sections_repr}"

    @property
    def model_name(self) -> str:
        """
        Construct and return the model name using meta configuration.

        Returns:
            str: The model name in the format '<meta.name>_v<meta.version>'.
        """
        return f"{self.meta.name}_v{self.meta.version}"
