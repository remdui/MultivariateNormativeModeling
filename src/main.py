"""Entry point for the software."""

from config.config_manager import ConfigManager
from entities.properties import Properties
from training.train import train_vae
from util.cmd_utils import parse_args
from util.config_utils import create_default_config
from util.log_utils import log_message, write_output


def run_training():
    """Run the training process."""
    print("Starting training")
    train_vae()


def run_validation():
    """Run the validation process."""
    print("Starting validation")
    # Retrieve the Properties object
    properties = Properties.get_instance()

    # Get the log and output directories
    log_dir = properties.system.log_dir
    output_dir = properties.system.output_dir
    model_name = properties.meta.name + "_v" + str(properties.meta.model_version)

    # Temporarily log a message and write an output file for demonstration purposes
    log_message("Validation started", log_dir)
    write_output(
        "Accuracy: " + "1.00", output_dir, model_name, "metrics", use_date=False
    )


def run_inference():
    """Run the inference process."""
    print("Starting inference")


if __name__ == "__main__":
    # Create a default configuration file
    create_default_config()

    # Parse command-line arguments
    args = parse_args()

    # Create ConfigManager instance
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Check if the configuration file is compatible with the current software version
    config_manager.is_version_compatible()

    # Retrieve the Properties object
    config = config_manager.get_config()

    # Initialize the Properties object with the merged configuration
    Properties.initialize(config)

    # Display the merged configuration
    log_message(str(Properties.get_instance()))

    # Perform action based on the argument
    if args.mode == "train":
        run_training()
    elif args.mode == "validate":
        run_validation()
    elif args.mode == "inference":
        if not args.checkpoint:
            raise ValueError(
                "For inference, you must provide a model checkpoint with --checkpoint"
            )
        run_inference()
