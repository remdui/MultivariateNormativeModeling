"""Tests for the ConfigManager class."""

import argparse

import pytest

from config.config_manager import ConfigManager
from util.config_utils import create_default_config

# Path to the default configuration file
DEFAULT_CONFIG_FILE = "config_default.yml"


@pytest.fixture(scope="session", autouse=True)
def setup_default_config():
    """Ensures the default config file is created before tests run."""
    create_default_config()


def test_load_default_config_sections():
    """Test that all sections are loaded correctly from the default configuration file."""

    config_manager = ConfigManager(
        config_file=DEFAULT_CONFIG_FILE, command_line_args=argparse.Namespace()
    )

    config = config_manager.get_config()

    # Verify that the main sections are present
    assert "dataset" in config
    assert "general" in config
    assert "meta" in config
    assert "model" in config
    assert "system" in config
    assert "train" in config
    assert "validation" in config


def test_override_with_command_line_args():
    """Test that command-line arguments override default configuration values."""
    # Simulated command-line arguments
    mock_args = argparse.Namespace(
        debug=True,
        log_level="DEBUG",
        train_split=0.6,
        test_split=0.25,
        val_split=0.15,
        epochs=50,
    )

    config_manager = ConfigManager(
        config_file=DEFAULT_CONFIG_FILE, command_line_args=mock_args
    )
    config = config_manager.get_config()

    # Check that specific arguments override the defaults
    assert config["general"]["debug"] is True
    assert config["general"]["log_level"] == "DEBUG"
    assert config["dataset"]["train_split"] == 0.6
    assert config["dataset"]["test_split"] == 0.25
    assert config["dataset"]["val_split"] == 0.15
    assert config["train"]["epochs"] == 50


def test_handle_missing_config_file():
    """Test behavior when configuration file is missing."""
    with pytest.raises(FileNotFoundError):
        ConfigManager(
            config_file="nonexistent_config.yml", command_line_args=argparse.Namespace()
        )


def test_no_command_line_args():
    """Test that configuration remains unchanged when no command-line arguments are provided."""
    config_manager = ConfigManager(
        config_file=DEFAULT_CONFIG_FILE, command_line_args=argparse.Namespace()
    )

    config = config_manager.get_config()

    # Check that all sections are still in the config as loaded from the file
    assert "dataset" in config
    assert "general" in config
    assert "meta" in config
    assert "model" in config
    assert "system" in config
    assert "train" in config
    assert "validation" in config

    # No overrides should have occurred
    assert config["general"]["debug"] is False
    assert config["train"]["epochs"] == 20  # Verify a default value
