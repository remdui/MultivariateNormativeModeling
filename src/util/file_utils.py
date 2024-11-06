"""Utility functions for logging."""

import os

from entities.log_manager import LogManager
from entities.properties import Properties


def write_output(
    output: str,
    output_identifier: str = "metrics",
    task: str = "train",
) -> None:
    """Write the output to the specified directory."""
    properties = Properties.get_instance()
    output_dir = properties.system.output_dir
    model_name = properties.model_name

    filename = f"{output_dir}/{model_name}_{task}_{output_identifier}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)


def create_storage_directories() -> None:
    """Create directories for storing logs and models if they do not exist.

    Data directory is not created here as it is required to be created by the user.
    """
    # Get logger
    logger = LogManager.get_logger(__name__)

    # Get file paths from properties
    properties = Properties.get_instance()
    log_dir = properties.system.log_dir
    models_dir = properties.system.models_dir
    output_dir = properties.system.output_dir

    # Create log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Created log directory: {log_dir}")

    # Create models directory if it does not exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")

        # Create checkpoints subdirectory
        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        os.makedirs(checkpoints_dir)
        logger.info(f"Created checkpoints directory: {checkpoints_dir}")

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")


def is_data_file(file_path: str) -> bool:
    """Check if the file is a data file."""
    valid_file_extensions = [
        ".csv",
        ".rds",
        ".parquet",
        ".xls",
        ".xlsx",
        ".feather",
        ".dta",
        ".json",
        ".txt",
        ".pkl",
    ]
    return any(file_path.endswith(extension) for extension in valid_file_extensions)


def is_image_folder(folder_path: str) -> bool:
    """Check if the provided path is a folder and check if the folder contains images in any of its child directories."""
    if not os.path.isdir(folder_path):
        return False

    for _, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                return True

    return False


def get_processed_file_path(data_dir: str, input_data: str, dataset_type: str) -> str:
    """Get the processed file path based on the input data."""
    input_file_name, _ = input_data.split(".")
    return os.path.join(data_dir, "processed", f"{input_file_name}_{dataset_type}.csv")
