"""Analysis helper functions for image data."""

import numpy as np
import PIL
from PIL.Image import Image
from torch import Tensor


def tensor_to_image(
    tensor: Tensor,
    image_length: int,
    image_width: int,
    slice_index: int | None = None,
) -> Image:
    """Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (Tensor): Input tensor of shape (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened.
        image_length (int): Image length
        image_width (int): Image width
        slice_index (int, optional): Index of the depth slice to visualize for 3D images. Defaults to the middle slice.

    Returns:
        Image: PIL image
    """

    # Ensure tensor is on CPU and detached from the computation graph
    tensor = tensor.cpu().detach()

    # Convert tensor to correct shape and format
    tensor = __convert_tensor(image_length, image_width, slice_index, tensor)

    # Scale to 0-255 and convert to uint8
    array = (tensor.numpy() * 255).astype(np.uint8)
    image_mode = "RGB" if array.ndim == 3 else "L"  # Choose mode based on array shape
    image = PIL.Image.fromarray(array, mode=image_mode)

    return image


def __convert_tensor(
    image_length: int, image_width: int, slice_index: int | None, tensor: Tensor
) -> Tensor:
    """Convert tensor to the correct shape and format for visualization.

    Args:
        image_length (int): Image length
        image_width (int): Image width
        slice_index (int, optional): Index of the depth slice to visualize for 3D images. Defaults to the middle slice.
        tensor (Tensor): Input tensor of shape (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened.

    Returns:
        Tensor: Processed tensor
    """

    # Convert (1, D, H, W) to (D, H, W) for single-channel 3D data
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Handle various tensor shapes
    if tensor.ndim == 4 and tensor.size(0) == 3:  # RGB 3D data (3, D, H, W)
        tensor = __convert_rgb_3d_tensor(tensor, slice_index)

    elif tensor.ndim == 3:
        if tensor.size(0) == 1:  # Grayscale 2D data (1, H, W)
            tensor = __convert_grayscale_2d_tensor(tensor)
        elif tensor.size(0) == 3:  # RGB 2D data (3, H, W)
            tensor = __convert_rgb_2d_tensor(tensor)
        else:  # Grayscale 3D data (D, H, W)
            tensor = __convert_grayscale_3d_tensor(tensor, slice_index)

    elif tensor.ndim == 2:  # Already in (H, W) format for grayscale 2D image
        pass  # No changes needed

    elif tensor.ndim == 1:  # Flattened data
        tensor = __convert_flattened_tensor(tensor, image_length, image_width)

    else:
        raise ValueError(
            f"Unsupported tensor shape {tensor.shape}. Expected (C, D, H, W), (D, H, W), (C, H, W), (H, W), or flattened."
        )
    return tensor


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


def __convert_rgb_3d_tensor(tensor: Tensor, slice_index: int | None) -> Tensor:
    """Process RGB 3D tensor to extract a specific slice and reorder dimensions."""
    slice_index = slice_index or tensor.size(1) // 2
    return tensor[:, slice_index, :, :].permute(1, 2, 0)


def __convert_grayscale_3d_tensor(tensor: Tensor, slice_index: int | None) -> Tensor:
    """Process Grayscale 3D tensor to extract a specific slice."""
    slice_index = slice_index or tensor.size(0) // 2
    return tensor[slice_index, :, :]


def __convert_grayscale_2d_tensor(tensor: Tensor) -> Tensor:
    """Process Grayscale 2D tensor by squeezing the singleton channel dimension."""
    return tensor.squeeze(0)


def __convert_rgb_2d_tensor(tensor: Tensor) -> Tensor:
    """Process RGB 2D tensor by permuting to (H, W, C)."""
    return tensor.permute(1, 2, 0)


def __convert_flattened_tensor(
    tensor: Tensor, image_length: int, image_width: int
) -> Tensor:
    """Reshape a flattened tensor to (H, W)."""
    return tensor.view(image_length, image_width)
