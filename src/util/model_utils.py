"""Model utilities."""

import os
from datetime import datetime

import torch
from torch import nn

from entities.log_manager import LogManager


def save_model(
    model: nn.Module,
    epoch: int,
    save_dir: str = "models",
    model_name: str = "vae_model",
    date_format: str = "%Y%m%d",
    use_date: bool = True,
) -> None:
    """Save the model to the specified directory."""
    logger = LogManager.get_logger(__name__)

    if model is None:
        raise ValueError("Model is invalid.")
    if epoch < 0:
        raise ValueError("Epoch cannot be negative.")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if use_date:
        date = datetime.now().strftime(date_format)
        filename = f"{save_dir}/{model_name}_{epoch}_{date}.pt"
    else:
        filename = f"{save_dir}/{model_name}_{epoch}.pt"

    torch.save(model, filename)
    logger.info(f"Saved model to: {filename}")
