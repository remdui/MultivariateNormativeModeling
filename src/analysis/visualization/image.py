"""Analysis helper functions for image data."""

import numpy as np
import PIL
import torch
from PIL.Image import Image


def tensor_to_image(
    tensor: torch.Tensor,
    image_length: int,
    image_width: int,
    slice_index: int | None = None,
) -> Image:
    """Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened.
        image_length (int): Image length
        image_width (int): Image width
        slice_index (int, optional): Index of the depth slice to visualize for 3D images. Defaults to the middle slice.

    Returns:
        Image: PIL image
    """

    # Ensure tensor is on CPU and detached from the computation graph
    tensor = tensor.cpu().detach()

    # Convert (1, D, H, W) to (D, H, W) for single-channel 3D data
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Handle various tensor shapes
    if tensor.ndim == 4 and tensor.size(0) == 3:  # RGB 3D data (3, D, H, W)
        slice_index = slice_index or tensor.size(1) // 2
        tensor = tensor[:, slice_index, :, :].permute(1, 2, 0)  # Reorder to (H, W, C)

    elif tensor.ndim == 3:
        if tensor.size(0) == 1:  # Grayscale 2D data (1, H, W)
            tensor = tensor.squeeze(0)
        elif tensor.size(0) == 3:  # RGB 2D data (3, H, W)
            tensor = tensor.permute(1, 2, 0)
        else:  # Grayscale 3D data (D, H, W)
            slice_index = slice_index or tensor.size(0) // 2
            tensor = tensor[slice_index, :, :]

    elif tensor.ndim == 2:  # Already in (H, W) format for grayscale 2D image
        pass  # No changes needed

    elif tensor.ndim == 1:  # Flattened data
        tensor = tensor.view(image_length, image_width)

    else:
        raise ValueError(
            f"Unsupported tensor shape {tensor.shape}. Expected (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened."
        )

    # Scale to 0-255 and convert to uint8
    array = (tensor.numpy() * 255).astype(np.uint8)
    image_mode = "RGB" if array.ndim == 3 else "L"  # Choose mode based on array shape
    image = PIL.Image.fromarray(array, mode=image_mode)

    return image


def combine_images(original_image: Image, reconstructed_image: Image) -> Image:
    """Combine original and reconstructed images side by side.

    Args:
        original_image (Image): Original image
        reconstructed_image (Image): Reconstructed image

    Returns:
        Image: Combined image
    """
    # Choose mode and size for the combined image based on input images
    mode = (
        original_image.mode
        if original_image.mode == reconstructed_image.mode
        else "RGB"
    )
    combined_width = original_image.width + reconstructed_image.width
    combined_height = max(original_image.height, reconstructed_image.height)

    combined_image = PIL.Image.new(mode, (combined_width, combined_height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(reconstructed_image, (original_image.width, 0))

    return combined_image
