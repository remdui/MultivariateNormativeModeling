"""Entry point for the application, handling initialization, configuration, and task execution."""

import logging
import time

from config.config_manager import ConfigManager
from entities.experiment_manager import ExperimentManager
from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.pipeline.factory import create_preprocessing_pipeline
from preprocessing.transform.impl.encoding import EncodingTransform
from tasks.experiment.experiment_task import ExperimentTask
from tasks.inference.inference_task import InferenceTask
from tasks.training.train_task import TrainTask
from tasks.tuning.tuning_task import TuningTask
from tasks.validation.validate_task import ValidateTask
from util.cmd_utils import parse_args
from util.config_utils import create_default_config
from util.file_utils import create_storage_directories
from util.system_utils import log_system_info

# Mapping of available tasks
TASK_MAP = {
    "train": TrainTask,
    "validate": ValidateTask,
    "inference": InferenceTask,
    "tune": TuningTask,
    "experiment": ExperimentTask,
}

# Parse command-line arguments
ARGS = parse_args()


def setup_basic_logging() -> None:
    """
    Sets up a basic logging configuration.

    This is used as a fallback before the application-specific logging setup is configured.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def initialize_application() -> None:
    """
    Initializes the application by setting up configuration, properties, and logging.

    This function:
    - Ensures a default configuration file exists.
    - Loads the configuration and validates compatibility.
    - Initializes the experiment and logging system.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If the configuration is incompatible.
        RuntimeError: If an unexpected error occurs during initialization.
    """
    try:
        create_default_config()

        # Load and validate configuration
        config_manager = ConfigManager(config_file=ARGS.config, command_line_args=ARGS)
        config_manager.is_version_compatible()
        config_manager.validate_config()

        # Initialize properties from the configuration
        Properties.initialize(config_manager.get_config())

        # Ensure necessary directories exist
        create_storage_directories()

        # Setup experiment directory
        experiment_manager = ExperimentManager.get_instance()
        experiment_manager.set_config_path(ARGS.config)

        # Reconfigure logging with the loaded properties
        LogManager.reconfigure_logging()

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise
    except ValueError as e:
        logging.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during initialization: {e}", exc_info=True)
        raise RuntimeError("Application initialization failed.") from e


def log_application_info() -> None:
    """Logs application metadata, system configuration, and environment details."""
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    logger.info(f"Application (v{properties.meta.version}) initialized successfully")
    logger.info(
        f"Using configuration file: {ARGS.config} (v{properties.meta.config_version})"
    )
    logger.info(f"Model/Experiment Name: {properties.meta.name}")
    logger.info(f"Description: {properties.meta.description}")

    # Log system and hardware details
    log_system_info()

    # Log detailed configuration
    logger.debug(properties)


def apply_preprocessing() -> None:
    """
    Executes the preprocessing pipeline based on the dataset type.

    Raises:
        RuntimeError: If preprocessing fails.
    """
    logger = logging.getLogger(__name__)
    try:
        properties = Properties.get_instance()
        pipeline = create_preprocessing_pipeline(properties.dataset.data_type)
        pipeline.run()
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise RuntimeError("Preprocessing pipeline execution failed.") from e


def run_task(task_class: type) -> None:
    """
    Executes a specified task such as training, validation, or inference.

    Args:
        task_class (type): The task class to be executed.

    Raises:
        RuntimeError: If the task execution fails.
    """
    logger = LogManager.get_logger(__name__)

    task_instance = task_class()
    task_name = task_instance.get_task_name()

    start_time = time.time()

    try:
        task_instance.run()
    except Exception as e:
        logger.error(f"Task '{task_name}' failed: {e}", exc_info=True)
        raise RuntimeError(f"Task '{task_name}' encountered an error.") from e

    runtime = time.time() - start_time
    logger.info(f"Task '{task_name}' completed in {runtime:.2f} seconds.")


def main() -> None:
    """
    Main function to control the application workflow.

    - Initializes the application and its components.
    - Executes preprocessing if enabled.
    - Runs the selected task based on command-line arguments.

    Raises:
        ValueError: If an invalid task mode is provided.
        RuntimeError: If any execution step fails.
    """
    setup_basic_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting application...")
        total_start_time = time.time()

        # Step 1: Application Initialization
        initialize_application()

        # Step 2: Log application details
        log_application_info()

        # Step 3: Execute Preprocessing Pipeline
        if not ARGS.skip_preprocessing:
            start_preprocessing_time = time.time()
            apply_preprocessing()
            logger.info(
                f"Preprocessing completed in {time.time() - start_preprocessing_time:.2f} seconds."
            )
        else:
            logger.info("Skipping preprocessing pipeline.")

        # Step 4: Run Selected Task
        task_class = TASK_MAP.get(ARGS.mode)
        if not task_class:
            logger.error(f"Invalid task mode specified: {ARGS.mode}")
            raise ValueError(
                f"Unsupported mode '{ARGS.mode}'. Available modes: {list(TASK_MAP.keys())}"
            )

        logger.info(f"Executing task: {ARGS.mode}")
        run_task(task_class)

        # Save normalization statistics
        EncodingTransform.save_stats_to_file()

        total_runtime = time.time() - total_start_time
        logger.info(f"Application completed in {total_runtime:.2f} seconds.")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
