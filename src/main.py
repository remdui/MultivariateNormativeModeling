from entities.properties import set_properties
from training.train import train_vae
from util.cmd_utils import parse_args
from config.config_manager import ConfigManager
from util.config_utils import create_default_config


def run_training():
    print("Starting training")
    train_vae()

def run_validation():
    print("Starting validation")

def run_inference():
    print("Starting inference")


if __name__ == "__main__":
    # Create a default configuration file
    create_default_config()

    # Parse command-line arguments
    args = parse_args()

    # Create ConfigManager instance
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Retrieve the Properties object
    properties = config_manager.get_properties()

    # Display the merged configuration
    print(properties)

    # Set properties globally
    set_properties(properties)


    # Perform action based on the argument
    if args.action == 'train':
        run_training()
    elif args.action == 'validate':
        run_validation()
    elif args.action == 'inference':
        if not args.checkpoint:
            raise ValueError("For inference, you must provide a model checkpoint with --checkpoint")
        run_inference()
