"""Utility functions for logging."""

import json
import os
from typing import Any, Literal

import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.task_result import TaskResult


def write_results_to_file(
    task_result: TaskResult,
    output_identifier: str = "metrics",
    task: Literal["train", "validate", "inference"] = "train",
) -> None:
    """Write the TaskResult content to a JSON file.

    Args:
        task_result (TaskResult): The TaskResult object containing the data to output.
        output_identifier (str): Identifier for the output, used in the filename.
        task (str): The task type (train, validate, or inference).
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()
    output_dir = properties.system.output_dir
    model_name = properties.model_name

    # Define the filename
    filename = f"{output_dir}/metrics/{model_name}_{task}_{output_identifier}.json"

    # Convert TaskResult data to a dictionary
    result_data = task_result.get_data()

    # Write the dictionary to a JSON file with indentation for readability
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Results written to {filename}")


def create_storage_directories() -> None:
    """Create directories for storing logs, models, and outputs if they do not exist.

    Data directory is required to be created by the user.
    """
    # Get logger and properties
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    # Define the main directories and their respective subdirectories
    directories = {
        properties.system.log_dir: [],
        properties.system.models_dir: ["checkpoints"],
        properties.system.output_dir: [
            "reconstructions",
            "visualizations",
            "model_arch",
            "metrics",
        ],
    }

    # Create each main directory and its subdirectories as needed
    for main_dir, sub_dirs in directories.items():
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            logger.info(f"Created directory: {main_dir}")

        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(main_dir, sub_dir)
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path)
                logger.info(f"Created subdirectory: {sub_dir_path}")


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


def save_data(data: pd.DataFrame, output_file_path: str) -> None:
    """Save the data to a file.

    HDF is the internal format used for data.
    """
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "csv":
        data.to_csv(output_file_path, index=False)
    elif file_format == "hdf":
        # Convert column names to strings to avoid issues with integer column names
        data.columns = data.columns.map(str)
        data.to_hdf(
            output_file_path,
            key="df",
            mode="w",
            complevel=4,
            complib="blosc",
            format="fixed",
            index=False,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_data(data_file_path: str) -> Any:
    """Load the data from a file.

    HDF is the internal format used for data.
    """
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "csv":
        return pd.read_csv(data_file_path)
    if file_format == "hdf":
        return pd.read_hdf(data_file_path, key="df")
    raise ValueError(f"Unsupported file format: {file_format}")


def is_image_folder(folder_path: str) -> bool:
    """Check if the provided path is a folder and check if the folder contains images in any of its child directories."""
    if not os.path.isdir(folder_path):
        return False

    for _, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                return True

    return False


def get_internal_file_extension() -> str:
    """Get the internal file extension based on the internal file format."""
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "hdf":
        return "h5"
    if file_format == "csv":
        return "csv"
    raise ValueError(f"Unsupported file format: {file_format}")


def get_processed_file_path(data_dir: str, input_data: str, dataset_type: str) -> str:
    """Get the processed file path based on the input data."""
    file_extension = get_internal_file_extension()

    input_file_name, _ = input_data.split(".")
    return os.path.join(
        data_dir, "processed", f"{input_file_name}_{dataset_type}.{file_extension}"
    )
