"""Model utilities."""

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

    properties = Properties.get_instance()
    file_format = properties.train.save_format

    if file_format == "pt":
        file_name += ".pt"
        torch.save(model.state_dict(), file_name)
    elif file_format == "safetensors":
        file_name += ".safetensors"
        st_save_model(model, file_name)
    else:
        raise ValueError(f"Model file format not supported: {file_format}")

    logger.info(f"Model state saved to: {file_name}")


def load_model(model: AbstractModel, model_path: str, device: str) -> AbstractModel:
    """Load a PyTorch state dict from a file into the model."""
    logger = LogManager.get_logger(__name__)

    if model_path.endswith(".pt"):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    if model_path.endswith(".safetensors"):
        st_load_model(model, model_path)
    else:
        raise ValueError(f"Model file extension not supported: {model_path}")

    model.to(device)
    model.eval()
    logger.info(f"Model state loaded from: {model_path}")
    return model


def visualize_model_arch(model: nn.Module, input_size: int) -> None:
    """Visualize the model architecture."""
    logger = LogManager.get_logger(__name__)

    properties = Properties.get_instance()
    batch_size = properties.train.batch_size
    output_dir = properties.system.output_dir + "/model_arch"

    file_name = "model_arch"

    # Check if the GPU supports CUDA compilation, torchview does not support JIT for all GPUs
    if gpu_supported_by_triton_compiler():
        logger.warning("Visualizing model architecture is not supported for this GPU.")
        return

    num_covariates = len(properties.dataset.covariates) - len(
        properties.dataset.skipped_covariates
    )

    # Draw the model architecture and save to output directory
    draw_graph(
        model,
        input_size=(batch_size, input_size),
        graph_name=file_name,
        save_graph=True,
        directory=output_dir,
        filename=file_name,
        covariates=torch.empty((batch_size, num_covariates)).to(
            properties.system.device
        ),
    )

    logger.info(
        f"Visualized model architecture and saved to: {output_dir}/{file_name}.png"
    )
