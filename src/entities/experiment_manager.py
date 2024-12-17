"""This module contains the ExperimentManager class."""

import os

from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import copy_artifact, get_processed_file_path, save_zip_folder


class ExperimentManager:
    """The ExperimentManager class manages the experiment directory and artifacts."""

    _instance = None

    def __init__(self) -> None:
        """Initialize the ExperimentManager instance."""
        self._experiment_path: str | None = None
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

    @classmethod
    def get_instance(cls) -> "ExperimentManager":
        """Return the ExperimentManager instance."""
        if cls._instance is None:
            cls._instance = ExperimentManager()
        return cls._instance

    def set_experiment_path(self, path: str) -> None:
        """Set the experiment path."""
        self._experiment_path = path

    def get_experiment_path(self) -> str | None:
        """Return the experiment path."""
        if self._experiment_path is None:
            raise ValueError(
                "Experiment path not set. Make sure to call set_experiment_path first."
            )
        return self._experiment_path

    def save_input_artifacts(self, config_file: str) -> None:
        """Save the input artifact to the experiment directory."""
        # Copy the config file into the experiment directory
        config_path = os.path.join("./config", config_file)
        self._copy_artifact_to_experiment(config_path)

        # Save source code
        save_zip_folder(
            folder_path="./src",
            dest_dir=str(self._experiment_path),
            zip_name="source_code",
        )

        # Copy the project dependencies
        self._copy_artifact_to_experiment("./poetry.lock")
        self._copy_artifact_to_experiment("./pyproject.toml")

        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data

        # Copy raw data
        raw_data_path = os.path.join(data_dir, input_data)
        self._copy_artifact_to_experiment(raw_data_path)

        # Copy processed train and test data
        train_output_path = get_processed_file_path(data_dir, input_data, "train")
        test_output_path = get_processed_file_path(data_dir, input_data, "test")
        self._copy_artifact_to_experiment(train_output_path)
        self._copy_artifact_to_experiment(test_output_path)

    def save_output_artifacts(self) -> None:
        """Save the output artifact to the experiment directory."""
        # Copy the application.log
        log_path = os.path.join(self.properties.system.log_dir, "application.log")
        self._copy_artifact_to_experiment(log_path)

    def _copy_artifact_to_experiment(self, src_path: str) -> None:
        """Copy an artifact to the experiment directory."""
        copy_artifact(src_path, str(self._experiment_path))
