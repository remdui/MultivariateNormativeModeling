"""Model utilities."""

import os
from datetime import datetime

import torch


def save_model(
    model,
    epoch,
    save_dir="models",
    model_name="vae_model",
    date_format="%Y%m%d",
    use_date=True,
):
    """Save the model to the specified directory."""
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
