"""Command line utility functions for the VAE pipeline."""

import argparse

from entities.log_manager import LogManager


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the VAE pipeline.

    This function sets up an argument parser, defines required and optional
    arguments, and returns the parsed arguments as an argparse.Namespace object.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.

    Command line arguments:
        --mode: Action to perform; choices are "train", "validate", "inference", or "tune".
        --config: Path to the configuration file.
        --device: Device to use (e.g., "cpu" or "cuda").
        --num_workers: Number of workers for data loading.
        --checkpoint: Path to the model checkpoint (for inference or resuming training).
        --checkpoint_interval: Interval for saving checkpoints.
        --log_dir: Directory to save logs and outputs.
        --output_dir: Directory to store inference results.
        --models_dir: Directory to save models.
        --data_dir: Directory to store data.
        --seed: Random seed.
        --log_level: Logging level; choices are "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        --verbose: Enable verbose output.
        --debug: Enable debug mode.
        --skip-preprocessing: If set, skip the preprocessing stage.
    """
    logger = LogManager.get_logger(__name__)

    parser = argparse.ArgumentParser(description="Run VAE Pipeline")

    # Required parameters
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "validate", "inference", "tune"],
        help="Action to perform: train, validate, inference, or tune",
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
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing stage",
    )

    args = parser.parse_args()
    logger.info("Command line arguments parsed successfully.")

    return args
