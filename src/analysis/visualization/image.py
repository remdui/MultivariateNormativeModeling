"""Analysis helper functions for image data."""

import numpy as np
import PIL
import torch
from PIL.Image import Image


def tensor_to_image(tensor: torch.Tensor, image_length: int, image_width: int) -> Image:
    """Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Input tensor
        image_length (int): Image length
        image_width (int): Image width

    Returns:
        Image: PIL image
    """

    # Ensure tensors are in shape (H, W) by reshaping or squeezing selectively
    if tensor.ndim == 3 and tensor.size(0) == 1:
        # For (1, H, W), we can safely squeeze out the singleton dimension
        tensor = tensor.squeeze(0)
    else:
        # For other shapes, reshape explicitly if necessary
        tensor = tensor.view(image_length, image_width)

    # Convert to PIL images
    image = PIL.Image.fromarray((tensor.numpy() * 255).astype(np.uint8))

    return image


def combine_images(original_image: Image, reconstructed_image: Image) -> Image:
    """Combine original and reconstructed images side by side.

    Args:
        original_image (Image): Original image
        reconstructed_image (Image): Reconstructed image

    Returns:
        Image: Combined image
    """
    combined_image = PIL.Image.new(
        "L", (original_image.width * 2, original_image.height)
    )
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(reconstructed_image, (original_image.width, 0))

    return combined_image
