"""Entry point for the application, handling initialization and control flow."""

import logging
import time
from argparse import Namespace

from config.config_manager import ConfigManager
from entities.experiment_manager import ExperimentManager
from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.pipeline.factory import create_preprocessing_pipeline
from tasks.inference.inference_task import InferenceTask
from tasks.training.train_task import TrainTask
from tasks.tuning.tuning_task import TuningTask
from tasks.validation.validate_task import ValidateTask
from util.cmd_utils import parse_args
from util.config_utils import create_default_config
from util.file_utils import (
    create_experiment_directory,
    create_storage_directories,
    write_results_to_file,
)
from util.model_utils import visualize_model_arch
from util.system_utils import log_system_info

# Task selection and execution
TASK_MAP = {
    "train": TrainTask,
    "validate": ValidateTask,
    "inference": InferenceTask,
    "tune": TuningTask,
}


def setup_basic_logging() -> None:
    """Sets up a basic temporary logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def initialize_application(args: Namespace) -> None:
    """Initializes the application by setting up configuration, properties, and logging.

    Args:
        args (Namespace): The command-line arguments.
    """
    try:
        # Create a default configuration file if it does not exist
        create_default_config()

        # Initialize the ConfigManager with the configuration file and command-line arguments
        config_manager = ConfigManager(config_file=args.config, command_line_args=args)

        # Check if the configuration file is compatible with the current software version
        config_manager.is_version_compatible()

        # Validate the configuration
        config_manager.validate_config()

        # Initialize the Properties object with the merged configuration
        config = config_manager.get_config()
        Properties.initialize(config)

        # Create storage directories if they do not exist
        create_storage_directories()

        # Setup experiment directory
        experiment_path = create_experiment_directory(args.mode)
        ExperimentManager.get_instance().set_experiment_path(experiment_path)

        # Reconfigure logging with the actual properties
        LogManager.reconfigure_logging()

    except Exception as e:
        logging.error(f"Error during application initialization: {e}", exc_info=True)
        raise


def log_application_info(args: Namespace) -> None:
    """Logs initial information about application and system.

    Args:
        args (Namespace): The command-line arguments.
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    # Display application version and experiment information
    logger.info(f"Application (v{properties.meta.version}) initialized successfully")
    logger.info(
        f"Configuration file: {args.config} (v{properties.meta.config_version})"
    )
    logger.info(f"Model/Experiment Name: {properties.meta.name}")
    logger.info(f"Description: {properties.meta.description}")

    # Log system information
    log_system_info()

    # Log the application properties
    logger.debug(properties)


def apply_preprocessing() -> None:
    """Apply the preprocessing pipeline to the input data."""
    logger = logging.getLogger(__name__)
    try:
        properties = Properties.get_instance()
        data_type = properties.dataset.data_type
        pipeline = create_preprocessing_pipeline(data_type)
        pipeline.run()
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


def run_task(task_class: type) -> None:
    """Generalized function to run a training, validation, or inference task.

    Args:
        task_class: The class of the task to execute (e.g., TrainTask, ValidateTask).
    """
    logger = LogManager.get_logger(__name__)
    task_instance = task_class()
    task_name = task_instance.get_task_name()

    start_time = time.time()

    try:
        # Run the task and process results
        results = task_instance.run()
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics", task_name)

        # If the task is train, visualize the model
        if task_name == "train":
            model = task_instance.get_model()
            visualize_model_arch(model, task_instance.get_input_size())

    except Exception as e:
        logger.error(f"Task {task_name} failed: {e}", exc_info=True)
        raise

    runtime = time.time() - start_time
    logger.info(f"Task {task_name} completed in {runtime:.2f} seconds.")


def main() -> None:
    """Main function to control application flow based on command-line arguments."""
    setup_basic_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading application...")
        start_time = time.time()

        # Parse command-line arguments
        args = parse_args()

        # Initialize the application
        initialize_application(args)

        # Reconfigure logger after full initialization
        logger = LogManager.get_logger(__name__)
        log_application_info(args)

        # Preprocessing pipeline
        preprocessing_runtime = 0.0
        if not args.skip_preprocessing:
            start_time_preprocessing = time.time()
            apply_preprocessing()
            preprocessing_runtime = time.time() - start_time_preprocessing
        else:
            logger.info("Skipping preprocessing pipeline.")

        # Save input artifacts to the experiment directory
        experiment_manager = ExperimentManager.get_instance()
        experiment_manager.save_input_artifacts(args.config)

        # Run the selected task
        task = TASK_MAP.get(args.mode)
        if task:
            logger.info(f"Starting {args.mode} process...")
            run_task(task)
        else:
            logger.error(f"Unimplemented mode specified: {args.mode}")
            return

        # Report runtime information
        if preprocessing_runtime > 0:
            logger.info(
                f"Preprocessing completed in {preprocessing_runtime:.2f} seconds."
            )
        total_runtime = time.time() - start_time
        logger.info(f"Application completed in {total_runtime:.2f} seconds.")

        # Save output artifacts to the experiment directory
        experiment_manager.save_output_artifacts()

    except Exception as e:
        logger.error(f"An unhandled error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
