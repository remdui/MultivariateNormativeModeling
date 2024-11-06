"""Model utilities."""

import os
from datetime import datetime

import torch
from torch import nn
from torchview import draw_graph

from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.abstract_model import AbstractModel


def save_model(
    model: nn.Module,
    epoch: int | None = None,
    save_dir: str = "models",
    model_name: str = "vae_model",
    use_date: bool = False,
    save_as_checkpoint: bool = False,
) -> None:
    """
    Save the model to the specified directory.

    Args:
        model (nn.Module): Model to save.

        epoch (Optional[int]): Epoch number; None for final save.
        save_dir (str): Directory to save the model.
        model_name (str): Base name for the model file.
        use_date (bool): Whether to include the current date in the file name.
        save_as_checkpoint (bool): If True, save as an intermediate checkpoint.
    """
    logger = LogManager.get_logger(__name__)
    if model is None:
        raise ValueError("Model cannot be None.")

    if save_as_checkpoint:
        save_dir = os.path.join(save_dir, "checkpoints")

    # Set up the file naming
    file_name = f"{save_dir}/{model_name}"

    # Add epoch number for checkpoints
    if epoch is not None:
        file_name += f"_{epoch}"

    # Add date if required
    if use_date:
        date_str = datetime.now().strftime("%Y%m%d")
        file_name += f"_{date_str}"

    # Finalize file name with extension
    file_name += ".pt"

    # Save only state dict to reduce file size
    torch.save(model.state_dict(), file_name)
    logger.info(f"Model state saved to: {file_name}")


def load_model(model: AbstractModel, model_path: str, device: str) -> AbstractModel:
    """Load a PyTorch state dict from a file into the model."""
    logger = LogManager.get_logger(__name__)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    logger.info(f"Model state loaded from: {model_path}")
    return model


def visualize_model(model: nn.Module, input_size: int) -> None:
    """Visualize the model architecture."""
    logger = LogManager.get_logger(__name__)

    properties = Properties.get_instance()
    batch_size = properties.train.batch_size
    output_dir = properties.system.output_dir
    model_name = properties.model_name

    file_name = f"{model_name}_model_arch"

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
