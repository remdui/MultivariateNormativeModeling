"""Utility functions for logging."""

import os
from datetime import datetime


def log_message(message, save_dir="logs"):
    """Log a message to the specified directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(
        f"{save_dir}/log_{datetime.now().strftime('%Y%m%d')}.txt", "a", encoding="utf-8"
    ) as f:
        f.write(message + "\n")


def write_output(
    output,
    save_dir="output",
    model_name="vae_model",
    output_identifier="metrics",
    date_format="%Y%m%d",
    use_date=True,
):
    """Write the output to the specified directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if use_date:
        date = datetime.now().strftime(date_format)
        filename = f"{save_dir}/{model_name}_{output_identifier}_{date}.txt"
    else:
        filename = f"{save_dir}/{model_name}_{output_identifier}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
