from training.train import train_vae
from util.cmd_utils import parse_args
from util.config_utils import ConfigManager


def train(train_properties):
    print("Starting training with config")
    train_vae(train_properties)

def validate(properties):
    print("Starting validation with config")

def inference(properties):
    print("Starting inference with config")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Create ConfigManager instance
    config_manager = ConfigManager(config_file=args.config, command_line_args=args)

    # Perform validity checks on parameters
    config_manager.check_validity()

    # Retrieve the Properties object
    properties = config_manager.get_properties()

    # Display the merged configuration
    properties.display()

    # Perform action based on the argument
    if args.action == 'train':
        train(properties)
    elif args.action == 'validate':
        validate(properties)
    elif args.action == 'inference':
        if not args.checkpoint:
            raise ValueError("For inference, you must provide a model checkpoint with --checkpoint")
        inference(properties)
