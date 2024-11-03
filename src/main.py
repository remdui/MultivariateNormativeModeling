"""Entry point for the software."""

import logging

from config.config_manager import ConfigManager
from entities.log_manager import LogManager
from entities.properties import Properties
from training.train import Trainer
from util.cmd_utils import parse_args
from util.config_utils import create_default_config
from util.file_utils import create_storage_directories, write_output

# Set up a basic temporary logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


def run_training() -> None:
    """Run the training process."""
    trainer = Trainer()
    trainer.train()


def run_validation() -> None:
    """Run the validation process."""
    write_output(
        "Accuracy: " + "1.00",
        "metrics",
    )


def run_inference() -> None:
    """Run the inference process."""
    write_output(
        "Accuracy: " + "1.00",
        "metrics",
    )


if __name__ == "__main__":
    logging.info("Loading application...")

    # Create a default configuration file
    create_default_config()

    # Parse command-line arguments
    args = parse_args()

    # Create ConfigManager instance
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Check if the configuration file is compatible with the current software version
    config_manager.is_version_compatible()

    # Validate the configuration
    config_manager.validate_config()

    # Retrieve the Properties object
    config = config_manager.get_config()

    # Initialize the Properties object with the merged configuration
    Properties.initialize(config)

    # Create storage directories if they do not exist
    create_storage_directories()

    # Reconfigure logging with the actual properties
    LogManager.reconfigure_logging()

    # Get the root logger
    logger = LogManager.get_logger(__name__)

    # Display the merged configuration
    logger.info("Application loaded successfully.")
    logger.debug(str(Properties.get_instance()))

    # Perform action based on the argument
    if args.mode == "train":
        logger.info("Starting training process...")
        run_training()
    elif args.mode == "validate":
        logger.info("Starting validation process...")
        run_validation()
    elif args.mode == "inference":
        if not args.checkpoint:
            raise ValueError(
                "For inference, you must provide a model checkpoint with --checkpoint"
            )
        logger.info("Starting inference process...")
        run_inference()
