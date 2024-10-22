from training.train import train_vae
from util.cmd_utils import parse_args
from config.config_manager import ConfigManager
from util.config_utils import create_default_config


def train(train_properties):
    print("Starting training")
    train_vae(train_properties)

def validate(properties):
    print("Starting validation")

def inference(properties):
    print("Starting inference")


if __name__ == "__main__":
    # Create a default configuration file
    create_default_config()

    # Parse command-line arguments
    args = parse_args()
    # Create ConfigManager instance
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Perform validity checks on parameters
    config_manager.check_validity()

    # Retrieve the Properties object
    properties = config_manager.get_properties()

    # Display the merged configuration
    print(properties)

    # Perform action based on the argument
    if args.action == 'train':
        train(properties)
    elif args.action == 'validate':
        validate(properties)
    elif args.action == 'inference':
        if not args.checkpoint:
            raise ValueError("For inference, you must provide a model checkpoint with --checkpoint")
        inference(properties)
