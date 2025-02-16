"""Experiment Manager."""

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
    """Manages experiment directory, input artifacts, and final output artifacts."""

    _instance = None

    def __init__(self) -> None:
        """Initialize the ExperimentManager instance."""
        self.config_path = ""
        self._experiment_path: str | None = None
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

    @classmethod
    def get_instance(cls) -> "ExperimentManager":
        """Get instance of the experiment manager."""
        if cls._instance is None:
            cls._instance = ExperimentManager()
        return cls._instance

    def clear_output_directory(self) -> None:
        """Remove all contents from output_dir before starting a new experiment if desired."""
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

        # Recreate storage directories if they are deleted.
        create_storage_directories()

    def create_new_experiment(self, task_name: str) -> None:
        """
        Creates a unique experiment directory under './experiments'.

        based on the provided task_name and the current timestamp.
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
        Appends a suffix to the existing experiment path.

        If the new path is different,
        it creates that folder (if missing) and updates the internal _experiment_path.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call set_experiment_path() first."
            )

        new_path = f"{self._experiment_path}_{suffix}"
        if new_path != self._experiment_path:
            os.makedirs(new_path, exist_ok=True)
            self._experiment_path = new_path
            self.logger.info(f"Experiment path updated to: {self._experiment_path}")

    def get_experiment_path(self) -> str:
        """Retrieve the current experiment path."""
        if not self._experiment_path:
            raise ValueError("Experiment path not set. Call set_experiment_path first.")
        return self._experiment_path

    def set_config_path(self, config_path: str) -> None:
        """Set the relative path to the config file (used in _save_input_artifacts)."""
        self.config_path = config_path

    def _save_input_artifacts(self) -> None:
        """
        Saves 'input' artifacts (config, dependencies, source code, data).

        in structured subfolders of the experiment folder:
            - config/
            - data/
            - source/
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call set_experiment_path() first."
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

        # 1. Config file
        if self.config_path:
            local_config_path = os.path.join("./config", self.config_path)
            copy_artifact(local_config_path, config_subdir)

        # 2. Lock files, project deps
        lock_file = "./poetry.lock"
        pyproject_file = "./pyproject.toml"
        if os.path.exists(lock_file):
            copy_artifact(lock_file, src_subdir)
        if os.path.exists(pyproject_file):
            copy_artifact(pyproject_file, src_subdir)

        # 3. Zip up source code into source/
        save_zip_folder("./src", src_subdir, zip_name="source_code")

        # 4. Data (raw + processed)
        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data

        raw_data_path = os.path.join(data_dir, input_data)
        if os.path.exists(raw_data_path):
            copy_artifact(raw_data_path, data_subdir)

        train_output_path = get_processed_file_path(data_dir, input_data, "train")
        test_output_path = get_processed_file_path(data_dir, input_data, "test")

        if os.path.exists(train_output_path):
            copy_artifact(train_output_path, data_subdir)
        if os.path.exists(test_output_path):
            copy_artifact(test_output_path, data_subdir)

    def _save_logs(self) -> None:
        """Copy the application.log from log_dir => experiment/logs/."""
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        # 1) Copy logs (only application.log)
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
        Copy model + checkpoints from models_dir => experiment/models/.

           - final model must be named 'self.properties.model_name' + .pt/.safetensors
           - checkpoint files start with 'self.properties.model_name_' + epoch + extension
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call create_new_experiment() first."
            )

        # 2) Copy model + checkpoints
        model_dir = self.properties.system.models_dir
        model_subdir = os.path.join(self._experiment_path, "models")
        os.makedirs(model_subdir, exist_ok=True)

        # 2a) Copy final model named e.g. my_model, ignoring extension
        possible_extensions = [".pt", ".safetensors"]
        for ext in possible_extensions:
            final_model_name = f"{self.properties.model_name}{ext}"
            final_model_src = os.path.join(model_dir, final_model_name)
            if os.path.exists(final_model_src):
                final_model_dest = os.path.join(model_subdir, final_model_name)
                shutil.copy2(final_model_src, final_model_dest)
                self.logger.info(
                    f"Copied final model {final_model_src} -> {final_model_dest}"
                )

    def _save_output_artifacts(self) -> None:
        """
        Copies subfolders from 'output_dir' into structured subfolders of the experiment path.

        Adjust `logical_folders` as needed to suit your pipeline's output structure.
        """
        if not self._experiment_path:
            raise ValueError(
                "Experiment path not set. Call set_experiment_path() first."
            )

        output_dir = self.properties.system.output_dir
        if not os.path.exists(output_dir):
            self.logger.warning(f"Output directory '{output_dir}' does not exist.")
            return

        self.logger.info(f"Copying contents from '{output_dir}' to experiment folder.")

        exp_subfolder_path = os.path.join(self._experiment_path, "output")
        os.makedirs(exp_subfolder_path, exist_ok=True)

        for subf in (
            "metrics",
            "reconstructions",
            "visualizations",
            "model_arch",
            "model",
        ):
            src_path = os.path.join(output_dir, subf)
            if os.path.exists(src_path):
                dest_path = os.path.join(exp_subfolder_path, subf)
                try:
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    self.logger.info(f"Copied {src_path} -> {dest_path}")
                except OSError as e:
                    self.logger.error(f"Error copying {src_path} to {dest_path}: {e}")

    def finalize_experiment(self) -> None:
        """
        A single combined method that does:

         1) save_input_artifacts()
         2) copy_output_to_experiment()
        """
        self._save_input_artifacts()
        self._save_logs()
        self._save_model_artifacts()
        self._save_output_artifacts()
        self.logger.info(
            "Experiment finalization completed (output copied & artifacts saved)."
        )
