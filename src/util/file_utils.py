"""
Utility functions for logging, file operations, and data management.

This module provides helper functions to write logs and results,
copy files while preserving directory structure if needed, create
experiment/storage directories, compress folders, and handle data
loading/saving with internal file formats.
"""

import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.task_result import TaskResult
from util.errors import UnsupportedFileFormatError


class NumpyEncoder(json.JSONEncoder):
    """Json encoder for numpy objects."""

    def default(self, o: Any) -> Any:
        """Convert numpy objects to native python objects."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        return super().default(o)


def write_results_to_file(
    task_result: TaskResult, output_identifier: str = "metrics"
) -> None:
    """
    Write TaskResult data to a JSON file.

    The function serializes the data from the TaskResult object into a
    human-readable JSON file. The filename is determined by the output
    directory defined in the system properties and the output_identifier.

    Args:
        task_result (TaskResult): Contains the data to be output.
        output_identifier (str): Identifier used in the filename (default "metrics").
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()
    output_dir = properties.system.output_dir

    # Define the output filename within the "metrics" subfolder.
    filename = os.path.join(output_dir, "metrics", f"{output_identifier}.json")

    # Serialize TaskResult data into a dictionary.
    result_data = task_result.get_data()

    # Write data to file with indentation for readability.
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, cls=NumpyEncoder)

    logger.info(f"Results written to {filename}")


def copy_artifact(
    src_path: str,
    dest_dir: str,
    preserve_structure: bool = False,
    base_dir: str | None = None,
) -> None:
    """
    Copy a file (artifact) from src_path to a destination directory.

    When preserve_structure is True, the function replicates the relative
    directory structure (computed from base_dir) in the destination directory.

    Args:
        src_path (str): Full path of the source file.
        dest_dir (str): Destination directory where the file will be copied.
        preserve_structure (bool): If True, replicates the file's relative
            path from base_dir under dest_dir. Defaults to False.
        base_dir (str | None): Base directory used to compute the relative path
            when preserve_structure is True. Must be provided if preserve_structure is True.

    Raises:
        ValueError: If preserve_structure is True but base_dir is not provided.
    """
    logger = LogManager.get_logger(__name__)

    if not os.path.exists(src_path):
        logger.warning(f"Source path does not exist and cannot be copied: {src_path}")
        return

    if os.path.isdir(src_path):
        logger.warning(
            f"Source path is a directory; copy_artifact() supports files only: {src_path}"
        )
        return

    # Compute destination path based on structure preservation.
    if preserve_structure:
        if base_dir is None:
            raise ValueError("base_dir must be provided when preserve_structure=True.")
        rel_dir = os.path.relpath(os.path.dirname(src_path), base_dir)
        dest_dir = os.path.join(dest_dir, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    else:
        # Ensure the destination directory exists.
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))

    shutil.copy(src_path, dest_path)
    logger.info(f"Copied artifact: {src_path} -> {dest_path}")


def create_experiment_directory(path: str) -> None:
    """
    Create an experiment directory using a specific naming convention.

    The experiment directory is created under the main experiments folder.
    The naming convention should follow the pattern:
        "<task>_<model_name>_<date>_<time>"

    Args:
        path (str): Full path to the experiment directory to be created.
    """
    properties = Properties.get_instance()

    # Ensure the main experiments directory exists.
    experiments_dir = properties.system.experiment_dir
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(path, exist_ok=True)


def save_zip_folder(folder_path: str, dest_dir: str, zip_name: str) -> None:
    """
    Compress the contents of a folder into a zip archive.

    The zip archive will be saved in the specified destination directory
    with the given base name.

    Args:
        folder_path (str): Path to the folder that will be zipped.
        dest_dir (str): Directory where the zip archive will be stored.
        zip_name (str): Base name for the zip file (extension will be added automatically).
    """
    logger = LogManager.get_logger(__name__)

    if not os.path.isdir(folder_path):
        logger.warning(f"Folder {folder_path} does not exist. Cannot zip.")
        return

    # Create the zip archive.
    archive_path = shutil.make_archive(
        base_name=os.path.join(dest_dir, zip_name),
        format="zip",
        root_dir=os.path.dirname(folder_path),
        base_dir=os.path.basename(folder_path),
    )

    logger.info(f"Zipped folder '{folder_path}' to '{archive_path}'.")


def create_storage_directories() -> None:
    """
    Create required storage directories for logs, models, and outputs.

    This function ensures that the main directories and their specific
    subdirectories exist. Note that the base data directory should be created
    by the user if required.
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    # Mapping of main directories to their subdirectories.
    directories = {
        properties.system.log_dir: [],
        properties.system.models_dir: ["checkpoints"],
        properties.system.output_dir: [
            "reconstructions",
            "model",
            "visualizations",
            "model_arch",
            "metrics",
        ],
    }

    # Create directories if they do not exist.
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
    """
    Determine whether the given file path corresponds to a data file.

    Recognized data file extensions include CSV, RDS, Parquet, Excel,
    Feather, Stata, JSON, TXT, and Pickle formats.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if the file has a recognized data file extension, else False.
    """
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
    """
    Save a pandas DataFrame to a file using the internal file format.

    The internal file format is defined in the system properties. Supported
    formats include CSV and HDF. When using HDF, column names are converted
    to strings to prevent potential issues.

    Args:
        data (pd.DataFrame): The data to be saved.
        output_file_path (str): The full path to the output file.

    Raises:
        UnsupportedFileFormatError: If the internal file format is unsupported.
    """
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "csv":
        data.to_csv(output_file_path, index=False)
    elif file_format == "hdf":
        # Ensure column names are strings (avoids issues with integer column names).
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
        raise UnsupportedFileFormatError(f"Unsupported file format: {file_format}")


def load_data(data_file_path: str) -> Any:
    """
    Load data from a file into a pandas DataFrame using the internal format.

    The internal file format is specified in the system properties and can
    be either CSV or HDF.

    Args:
        data_file_path (str): Full path to the data file.

    Returns:
        Any: A pandas DataFrame containing the loaded data.

    Raises:
        UnsupportedFileFormatError: If the internal file format is unsupported.
    """
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "csv":
        return pd.read_csv(data_file_path)
    if file_format == "hdf":
        return pd.read_hdf(data_file_path, key="df")
    raise UnsupportedFileFormatError(f"Unsupported file format: {file_format}")


def is_image_folder(folder_path: str) -> bool:
    """
    Check whether the specified folder contains image files.

    The function recursively searches through the folder's child directories
    for files with common image extensions (jpg, jpeg, png).

    Args:
        folder_path (str): Path to the folder to be checked.

    Returns:
        bool: True if at least one image file is found, otherwise False.
    """
    if not os.path.isdir(folder_path):
        return False

    for _, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                return True

    return False


def get_internal_file_extension() -> str:
    """
    Retrieve the file extension associated with the internal file format.

    The internal format is defined in the system properties and currently
    supports 'hdf' (maps to 'h5') and 'csv' formats.

    Returns:
        str: The file extension without the dot.

    Raises:
        UnsupportedFileFormatError: If the internal file format is unsupported.
    """
    properties = Properties.get_instance()
    file_format = properties.dataset.internal_file_format

    if file_format == "hdf":
        return "h5"
    if file_format == "csv":
        return "csv"
    raise UnsupportedFileFormatError(f"Unsupported file format: {file_format}")


def get_processed_file_path(data_dir: str, input_data: str, dataset_type: str) -> str:
    """
    Construct a file path for processed data.

    The function combines the base data directory, a 'processed' subdirectory,
    the input file name (without extension), and the dataset type, appending
    the appropriate internal file extension.

    Args:
        data_dir (str): Base directory for data storage.
        input_data (str): Input data filename (expected to include an extension).
        dataset_type (str): A descriptor for the dataset type.

    Returns:
        str: The full path for the processed data file.
    """
    file_extension = get_internal_file_extension()
    input_file_name = os.path.splitext(input_data)[0]
    return os.path.join(
        data_dir, "processed", f"{input_file_name}_{dataset_type}.{file_extension}"
    )
