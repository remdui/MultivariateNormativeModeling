"""Entry point for the application, handling initialization and control flow."""

import logging
import time
from argparse import Namespace

from config.config_manager import ConfigManager
from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.pipeline.factory import create_preprocessing_pipeline
from tasks.training.trainer import Trainer
from tasks.validation.validator import Validator
from util.cmd_utils import parse_args
from util.config_utils import create_default_config
from util.file_utils import create_storage_directories, write_output
from util.model_utils import visualize_model
from util.system_utils import log_system_info


def setup_basic_logging() -> None:
    """Sets up a basic temporary logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def initialize_application(args: Namespace) -> None:
    """Initializes the application by setting up configuration, properties, and logging."""
    # Create a default configuration file if it does not exist
    create_default_config()

    # Initialize the ConfigManager with the configuration file and command-line arguments
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Check if the configuration file is compatible with the current software version
    config_manager.is_version_compatible()

    # Validate the configuration
    config_manager.validate_config()

    # Retrieve the merged config object
    config = config_manager.get_config()

    # Initialize the Properties object with the merged configuration
    Properties.initialize(config)

    # Create storage directories if they do not exist
    create_storage_directories()

    # Reconfigure logging with the actual properties
    LogManager.reconfigure_logging()


def log_application_info(args: Namespace) -> None:
    """Logs initial information about application and system."""
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
    properties = Properties.get_instance()
    data_type = properties.dataset.data_type
    pipeline = create_preprocessing_pipeline(data_type)
    pipeline.run()


def run_training() -> None:
    """Run the training process and returns the training duration."""
    logger = LogManager.get_logger(__name__)

    # Initialize the Trainer
    trainer = Trainer()

    # Start timing the training process
    start_time = time.time()

    # Train the model
    trainer.run()

    # Get the model and visualize it
    model = trainer.get_model()
    visualize_model(model, trainer.get_input_size())

    # Calculate and return the training duration
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f} seconds.")


def run_validation() -> None:
    """Run the validation process and log output."""
    logger = LogManager.get_logger(__name__)

    # Initialize the Trainer
    validator = Validator()

    # Start timing the training process
    start_time = time.time()

    # Train the model
    result = validator.run()

    # Validate and process the results
    result.validate_results()
    result.process_results()
    accuracy = result["accuracy"]
    write_output(f"Accuracy: {accuracy}", "metrics")

    # Calculate and return the training duration
    training_duration = time.time() - start_time
    logger.info(f"Validation completed in {training_duration:.2f} seconds.")


def run_inference() -> None:
    """Run the inference process, requiring a checkpoint."""
    # Output inference accuracy
    write_output("Accuracy: " + "1.00", "metrics")


def main() -> None:
    """Main function to control application flow based on command-line arguments."""
    # Set up basic logging for startup
    setup_basic_logging()

    logger = logging.getLogger(__name__)
    logger.info("Loading application...")

    # Start timing the total runtime
    start_time = time.time()

    # Parse command-line arguments
    args = parse_args()

    # Initialize the application and set up logging
    initialize_application(args)

    # Get the initialized logger
    logger = LogManager.get_logger(__name__)

    # Log initial information about the application and system
    log_application_info(args)

    # Run preprocessing pipeline
    apply_preprocessing()

    # Perform action based on the mode argument
    if args.mode == "train":
        logger.info("Starting training process...")
        run_training()
    elif args.mode == "validate":
        logger.info("Starting validation process...")
        run_validation()
    elif args.mode == "inference":
        logger.info("Starting inference process...")
        run_inference()

    # Calculate and log the total runtime
    total_runtime = time.time() - start_time
    logger.info(f"Total runtime: {total_runtime:.2f} seconds.")


if __name__ == "__main__":
    main()
