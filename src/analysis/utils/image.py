"""Image tabular representation to tensor representation utils."""

import numpy as np
import PIL.Image
from PIL.Image import Image
from torch import Tensor


def tensor_to_image(
    tensor: Tensor, image_length: int, image_width: int, slice_index: int | None = None
) -> Image:
    """
    Convert a PyTorch tensor to a PIL Image.

    This function converts an input tensor (which can represent 2D or 3D image data)
    into a PIL Image. For 3D tensors, a specific slice is extracted for visualization.
    The tensor values are scaled to the 0-255 range and cast to uint8.

    Args:
        tensor (Tensor): Input tensor. Expected shapes include:
                         - (C, D, H, W) for 3D images with multiple channels,
                         - (D, H, W) for single-channel 3D images,
                         - (C, H, W) for 2D images with multiple channels,
                         - (H, W) for 2D images,
                         - or a flattened tensor.
        image_length (int): The target image height.
        image_width (int): The target image width.
        slice_index (int, optional): Index of the depth slice to visualize in 3D data.
                                     Defaults to the middle slice if not provided.

    Returns:
        Image: A PIL Image object.
    """
    # Ensure the tensor is on CPU and detached from computation.
    tensor = tensor.cpu().detach()
    # Reshape and adjust tensor dimensions as necessary.
    tensor = __convert_tensor(image_length, image_width, slice_index, tensor)
    # Scale tensor values to 0-255 and convert to unsigned 8-bit integers.
    array = (tensor.numpy() * 255).astype(np.uint8)
    # Choose image mode based on array dimensions: RGB for 3 channels, L for single channel.
    image_mode = "RGB" if array.ndim == 3 else "L"
    return PIL.Image.fromarray(array, mode=image_mode)


def __convert_tensor(
    image_length: int, image_width: int, slice_index: int | None, tensor: Tensor
) -> Tensor:
    """
    Convert a tensor to the appropriate shape for image visualization.

    This helper function adjusts the tensor shape depending on its dimensions:
      - Squeezes single-channel 3D tensors,
      - Extracts a slice from 3D tensors if needed,
      - Permutes dimensions for RGB 2D images,
      - Reshapes flattened tensors.

    Args:
        image_length (int): Target image height.
        image_width (int): Target image width.
        slice_index (int, optional): Depth slice index for 3D data; defaults to the middle slice.
        tensor (Tensor): The input tensor.

    Returns:
        Tensor: Processed tensor ready for conversion to a PIL image.

    Raises:
        ValueError: If the tensor shape is unsupported.
    """
    # Convert (1, D, H, W) to (D, H, W) for single-channel 3D data.
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Handle various tensor shapes.
    if tensor.ndim == 4 and tensor.size(0) == 3:  # RGB 3D data: (3, D, H, W)
        tensor = __convert_rgb_3d_tensor(tensor, slice_index)
    elif tensor.ndim == 3:
        if tensor.size(0) == 1:  # Grayscale 2D data: (1, H, W)
            tensor = __convert_grayscale_2d_tensor(tensor)
        elif tensor.size(0) == 3:  # RGB 2D data: (3, H, W)
            tensor = __convert_rgb_2d_tensor(tensor)
        else:  # Grayscale 3D data: (D, H, W)
            tensor = __convert_grayscale_3d_tensor(tensor, slice_index)
    elif tensor.ndim == 2:
        # Tensor is already in (H, W) format.
        pass
    elif tensor.ndim == 1:
        tensor = __convert_flattened_tensor(tensor, image_length, image_width)
    else:
        raise ValueError(
            f"Unsupported tensor shape {tensor.shape}. Expected (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened."
        )
    return tensor


def combine_images(original_image: Image, reconstructed_image: Image) -> Image:
    """
    Combine two images side by side.

    Places the original image and the reconstructed image next to each other,
    creating a new image with a width equal to the sum of both widths and a height
    equal to the maximum height of the two images.

    Args:
        original_image (Image): The original image.
        reconstructed_image (Image): The reconstructed image.

    Returns:
        Image: The combined image.
    """
    # Use the mode of the first image if both have the same mode; otherwise default to "RGB".
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


def __convert_rgb_3d_tensor(tensor: Tensor, slice_index: int | None) -> Tensor:
    """
    Extract a 2D RGB slice from a 3D RGB tensor.

    Args:
        tensor (Tensor): Input tensor of shape (3, D, H, W).
        slice_index (int, optional): Depth slice index; defaults to the middle slice if not provided.

    Returns:
        Tensor: Tensor of shape (H, W, 3) representing the extracted 2D RGB image.
    """
    # Determine the slice index: use provided value or default to middle slice.
    slice_index = slice_index or tensor.size(1) // 2
    # Extract the slice and permute dimensions to (H, W, C).
    return tensor[:, slice_index, :, :].permute(1, 2, 0)


def __convert_grayscale_3d_tensor(tensor: Tensor, slice_index: int | None) -> Tensor:
    """
    Extract a 2D grayscale slice from a 3D grayscale tensor.

    Args:
        tensor (Tensor): Input tensor of shape (D, H, W).
        slice_index (int, optional): Depth slice index; defaults to the middle slice if not provided.

    Returns:
        Tensor: Tensor of shape (H, W) representing the extracted 2D grayscale image.
    """
    slice_index = slice_index or tensor.size(0) // 2
    return tensor[slice_index, :, :]


def __convert_grayscale_2d_tensor(tensor: Tensor) -> Tensor:
    """
    Squeeze the singleton channel dimension from a 2D grayscale tensor.

    Args:
        tensor (Tensor): Input tensor of shape (1, H, W).

    Returns:
        Tensor: Tensor of shape (H, W).
    """
    return tensor.squeeze(0)


def __convert_rgb_2d_tensor(tensor: Tensor) -> Tensor:
    """
    Permute a 2D RGB tensor to have shape (H, W, C).

    Args:
        tensor (Tensor): Input tensor of shape (3, H, W).

    Returns:
        Tensor: Tensor of shape (H, W, 3).
    """
    return tensor.permute(1, 2, 0)


def __convert_flattened_tensor(
    tensor: Tensor, image_length: int, image_width: int
) -> Tensor:
    """
    Reshape a flattened tensor into a 2D image tensor.

    Args:
        tensor (Tensor): Flattened tensor.
        image_length (int): Target image height.
        image_width (int): Target image width.

    Returns:
        Tensor: Reshaped tensor of shape (image_length, image_width).
    """
    return tensor.view(image_length, image_width)
