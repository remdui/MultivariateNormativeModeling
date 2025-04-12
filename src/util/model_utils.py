"""Utilities for saving, loading, and visualizing PyTorch models."""

import os
from datetime import datetime

import torch
from safetensors.torch import load_model as st_load_model
from safetensors.torch import save_model as st_save_model
from torch import nn
from torchview import draw_graph

from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.abstract_model import AbstractModel
from util.system_utils import gpu_supported_by_triton_compiler


def save_model(
    model: nn.Module,
    epoch: int | None = None,
    save_dir: str = "models",
    model_name: str = "vae_model",
    use_date: bool = False,
    save_as_checkpoint: bool = False,
) -> None:
    """
    Saves a PyTorch model to disk in the specified format.

    Args:
        model (nn.Module): The PyTorch model to save.
        epoch (Optional[int]): Epoch number to include in the filename (for checkpoints).
        save_dir (str): Directory where the model should be saved.
        model_name (str): Base name of the model file.
        use_date (bool): If True, appends the current date to the filename.
        save_as_checkpoint (bool): If True, saves as an intermediate checkpoint in a subdirectory.

    Raises:
        ValueError: If an unsupported model file format is specified.
        OSError: If there is an issue saving the model file.
        TypeError: If the model instance is not a pytorch model object
    """
    logger = LogManager.get_logger(__name__)

    if not isinstance(model, nn.Module):
        raise TypeError("Expected 'model' to be an instance of nn.Module.")

    if save_as_checkpoint:
        save_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Construct file name
    file_name = f"{model_name}"
    if epoch is not None:
        file_name += f"_{epoch}"
    if use_date:
        file_name += f"_{datetime.now().strftime('%Y%m%d')}"

    properties = Properties.get_instance()
    file_format = properties.train.save_format

    try:
        if file_format == "pt":
            file_name += ".pt"
            torch.save(model.state_dict(), os.path.join(save_dir, file_name))
        elif file_format == "safetensors":
            file_name += ".safetensors"
            st_save_model(model, os.path.join(save_dir, file_name))
        else:
            raise ValueError(f"Unsupported model file format: {file_format}")

        logger.info(f"Model saved successfully: {os.path.join(save_dir, file_name)}")
    except OSError as e:
        logger.error(f"Failed to save model to {save_dir}: {e}")
        raise


def load_model(model: AbstractModel, model_path: str, device: str) -> AbstractModel:
    """
    Loads a model from disk and initializes it on the specified device.

    Args:
        model (AbstractModel): The model instance to load the state into.
        model_path (str): Path to the saved model file.
        device (str): The device ('cpu' or 'cuda') to load the model on.

    Returns:
        AbstractModel: The model with loaded state.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        ValueError: If the file format is unsupported.
        RuntimeError: If loading fails due to an invalid state dict.
    """
    logger = LogManager.get_logger(__name__)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_path.endswith(".pt"):
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif model_path.endswith(".safetensors"):
            st_load_model(model, model_path)
        else:
            raise ValueError(f"Unsupported model file format: {model_path}")

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from: {model_path}")
        return model

    except RuntimeError as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def visualize_model_arch(model: nn.Module, input_size: int, cov_size: int = 0) -> None:
    """
    Generates a visualization of the model architecture and saves it as an image.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        input_size (int): The input size for the visualization.
        cov_size (int): The size of the covariates tensor (if applicable).

    Raises:
        NotImplementedError: If visualization is not supported on the current GPU.
        OSError: If there is an issue saving the visualization file.
    """
    logger = LogManager.get_logger(__name__)

    properties = Properties.get_instance()
    batch_size = properties.train.batch_size
    output_dir = os.path.join(properties.system.output_dir, "model_arch")
    file_name = "model_arch"

    # Check if GPU visualization is supported
    if gpu_supported_by_triton_compiler():
        logger.warning("Model visualization is not supported on this GPU.")
        return

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    try:
        # Generate and save the model visualization
        draw_graph(
            model,
            input_size=(batch_size, input_size),
            graph_name=file_name,
            save_graph=True,
            directory=output_dir,
            filename=file_name,
            covariates=torch.empty((batch_size, cov_size)).to(properties.system.device),
        )
        logger.info(
            f"Model architecture visualization saved: {os.path.join(output_dir, file_name)}.png"
        )
    except OSError as e:
        logger.error(f"Failed to save model visualization: {e}")
        raise
