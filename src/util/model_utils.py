"""Model utilities."""

import os
from datetime import datetime

import torch
from torch import nn
from torchview import draw_graph

from entities.log_manager import LogManager
from entities.properties import Properties


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


def visualize_model(model: nn.Module, input_size: int) -> None:
    """Visualize the model architecture."""
    logger = LogManager.get_logger(__name__)

    properties = Properties.get_instance()
    batch_size = properties.train.batch_size
    output_dir = properties.system.output_dir
    model_name = properties.model_name

    file_name = f"model_arch_{model_name}"

    # Draw the model architecture and save to output directory
    draw_graph(
        model,
        input_size=(batch_size, input_size),
        graph_name=file_name,
        save_graph=True,
        directory=output_dir,
        filename=file_name,
    )

    logger.info(
        f"Visualized model architecture and saved to: {output_dir}/{file_name}.png"
    )
