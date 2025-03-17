"""Experiment Manager.

This module defines the ExperimentManager singleton, which is responsible for managing
experiment directories and handling input/output artifacts such as configuration files,
logs, model checkpoints, and output data.
"""

import os
import shutil
import time

from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import (
    copy_artifact,
    create_experiment_directory,
    create_storage_directories,
    get_processed_file_path,
    save_zip_folder,
)


class ExperimentManager:
    """
    Manages experiment directories, input artifacts, and output artifacts.

    This singleton class handles the creation of unique experiment directories, clearing
    output directories before a new experiment, and saving various artifacts such as configuration,
    source code, logs, and model checkpoints.
    """

    _instance = None

    def __init__(self) -> None:
        """Initialize the ExperimentManager instance."""
        self.config_path = ""
        self._experiment_path: str | None = None
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

    @classmethod
    def get_instance(cls) -> "ExperimentManager":
        """
        Retrieve the singleton ExperimentManager instance.

        Returns:
            ExperimentManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = ExperimentManager()
        return cls._instance

    def clear_output_directory(self) -> None:
        """
        Remove all contents from the output directory to ensure a clean state for a new experiment.

        After clearing, storage directories are recreated.
        """
        output_dir = self.properties.system.output_dir
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                path_item = os.path.join(output_dir, item)
                try:
                    if os.path.isdir(path_item):
                        shutil.rmtree(path_item)
                    else:
                        os.remove(path_item)
                except OSError as e:
                    self.logger.error(f"Error removing {path_item}: {e}")
        create_storage_directories()

    def create_new_experiment(self, task_name: str) -> None:
        """
        Create a new experiment directory under './experiments'.

        The directory name is constructed using the model name, task name, and current timestamp.

        Args:
            task_name (str): Identifier for the current task.
        """
        base_path = "./experiments"
        os.makedirs(base_path, exist_ok=True)

        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        model_name = self.properties.model_name
        folder_name = f"{model_name}_{task_name}_{timestamp}"
        new_path = os.path.join(base_path, folder_name)

        if new_path != self._experiment_path:
            os.makedirs(new_path, exist_ok=True)
            create_experiment_directory(new_path)
            self._experiment_path = new_path
            self.logger.info(f"Experiment directory created: {self._experiment_path}")

    def add_experiment_group_identifier(self, suffix: str) -> None:
        """
        Append a suffix to the current experiment path.

        Creates the new directory if it does not exist and updates the internal experiment path.

        Args:
            suffix (str): Suffix to append to the experiment path.

        Raises:
            ValueError: If the experiment path is not set.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )
        new_path = f"{self._experiment_path}_{suffix}"
        if new_path != self._experiment_path:
            os.makedirs(new_path, exist_ok=True)
            self._experiment_path = new_path
            self.logger.info(f"Experiment path updated to: {self._experiment_path}")

    def get_experiment_path(self) -> str:
        """
        Retrieve the current experiment directory path.

        Returns:
            str: The experiment directory path.

        Raises:
            ValueError: If the experiment path is not set.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )
        return self._experiment_path

    def set_config_path(self, config_path: str) -> None:
        """
        Set the relative path to the configuration file.

        This path is used when saving input artifacts.

        Args:
            config_path (str): Relative path to the configuration file.
        """
        self.config_path = config_path

    def _save_input_artifacts(self) -> None:
        """
        Save input artifacts (config, dependencies, source code, and data) to the experiment folder.

        Artifacts are organized into subfolders: config/, data/, and source/.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        self.logger.info(
            "Saving input artifacts (config, deps, src, data) to experiment."
        )

        config_subdir = os.path.join(self._experiment_path, "config")
        data_subdir = os.path.join(self._experiment_path, "data")
        src_subdir = os.path.join(self._experiment_path, "source")

        os.makedirs(config_subdir, exist_ok=True)
        os.makedirs(data_subdir, exist_ok=True)
        os.makedirs(src_subdir, exist_ok=True)

        # Save configuration file.
        if self.config_path:
            local_config_path = os.path.join("./config", self.config_path)
            copy_artifact(local_config_path, config_subdir)

        # Save dependency files.
        for file in ("./poetry.lock", "./pyproject.toml"):
            if os.path.exists(file):
                copy_artifact(file, src_subdir)

        # Zip and save source code.
        save_zip_folder("./src", src_subdir, zip_name="source_code")

        # Save raw and processed data.
        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data
        raw_data_path = os.path.join(data_dir, input_data)
        if os.path.exists(raw_data_path):
            copy_artifact(raw_data_path, data_subdir)

        for split in ("train", "test"):
            processed_path = get_processed_file_path(data_dir, input_data, split)
            if os.path.exists(processed_path):
                copy_artifact(processed_path, data_subdir)

    def _save_logs(self) -> None:
        """Copy the application log from the log directory to the experiment's logs folder."""
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        logs_dir = self.properties.system.log_dir
        log_src = os.path.join(logs_dir, "application.log")
        logs_subdir = os.path.join(self._experiment_path, "logs")
        os.makedirs(logs_subdir, exist_ok=True)

        if os.path.exists(log_src):
            log_dest = os.path.join(logs_subdir, "application.log")
            shutil.copy2(log_src, log_dest)
            self.logger.info(f"Copied {log_src} -> {log_dest}")
        else:
            self.logger.warning("No application.log found to copy.")

    def _save_model_artifacts(self) -> None:
        """
        Copy the final model and its checkpoints from the models directory to the experiment folder.

        The final model is expected to be named '{model_name}_best' with a supported extension (.pt or .safetensors).
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        model_dir = self.properties.system.models_dir
        model_subdir = os.path.join(self._experiment_path, "models")
        os.makedirs(model_subdir, exist_ok=True)

        possible_extensions = [".pt", ".safetensors"]
        for ext in possible_extensions:
            final_model_name = f"{self.properties.model_name}_best{ext}"
            final_model_src = os.path.join(model_dir, final_model_name)
            if os.path.exists(final_model_src):
                final_model_dest = os.path.join(model_subdir, final_model_name)
                shutil.copy2(final_model_src, final_model_dest)
                self.logger.info(
                    f"Copied final model {final_model_src} -> {final_model_dest}"
                )

    def _save_output_artifacts(self) -> None:
        """
        Copy output artifacts from the output directory into structured subfolders under the experiment folder.

        Expected subfolders include: metrics, reconstructions, visualizations, model_arch, and model.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        output_dir = self.properties.system.output_dir
        if not os.path.exists(output_dir):
            self.logger.warning(f"Output directory '{output_dir}' does not exist.")
            return

        self.logger.info(f"Copying contents from '{output_dir}' to experiment folder.")
        exp_subfolder_path = os.path.join(self._experiment_path, "output")
        os.makedirs(exp_subfolder_path, exist_ok=True)

        for subfolder in (
            "metrics",
            "reconstructions",
            "visualizations",
            "model_arch",
            "model",
        ):
            src_path = os.path.join(output_dir, subfolder)
            if os.path.exists(src_path):
                dest_path = os.path.join(exp_subfolder_path, subfolder)
                try:
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    self.logger.info(f"Copied {src_path} -> {dest_path}")
                except OSError as e:
                    self.logger.error(f"Error copying {src_path} to {dest_path}: {e}")

    def finalize_experiment(self) -> None:
        """
        Finalize the experiment by saving all input and output artifacts.

        This method saves input artifacts (config, dependencies, source code, and data),
        logs, model artifacts, and output artifacts.
        """
        self._save_input_artifacts()
        self._save_logs()
        self._save_model_artifacts()
        self._save_output_artifacts()
        self.logger.info(
            "Experiment finalization completed (artifacts saved and output copied)."
        )
