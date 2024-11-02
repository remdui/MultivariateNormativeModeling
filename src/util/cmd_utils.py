"""Command line utility functions."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VAE Pipeline")

    # Main parameters
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "validate", "inference"],
        help="Action to perform: train, validate, inference",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )

    # Optional parameters
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint (for inference or resuming training)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, help="Interval for saving checkpoints"
    )
    parser.add_argument(
        "--log_dir", type=str, help="Directory to save logs and outputs"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to store inference results"
    )
    parser.add_argument("--models_dir", type=str, help="Directory to save models")
    parser.add_argument("--data_dir", type=str, help="Directory to store data")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()
