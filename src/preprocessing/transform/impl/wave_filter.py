"""Module for filtering data based on wave value.

This module defines a PyTorch transform that filters rows in an input tensor by
comparing a specified column (containing wave information) to a target wave value.
If no target wave is specified (selected_wave == -1), the original data is returned.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class WaveFilterTransform(Transform):
    """
    PyTorch transform that filters input data based on a target wave value.

    This transform examines a specific column of the input tensor (designated by `col_id`)
    and returns only the rows where the wave value is approximately equal to the target
    value (`selected_wave`), within a small tolerance to account for floating-point precision.
    If `selected_wave` is -1, no filtering is applied.
    """

    def __init__(self, selected_wave: int = -1, col_id: int = 1) -> None:
        """
        Initialize the wave filter transform.

        Args:
            selected_wave (int): The target wave value for filtering. Use -1 to disable filtering.
            col_id (int): The index of the column in the input tensor that contains wave information.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.selected_wave = selected_wave
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """
        Filter the input tensor by the selected wave value.

        If `selected_wave` is -1, no filtering is performed and the original tensor is returned.

        Args:
            data (Tensor): Input tensor where one column holds wave information.

        Returns:
            Tensor: A tensor containing only the rows where the wave value matches the
            target value within a tolerance of 1e-5.
        """
        if self.selected_wave == -1:
            self.logger.info("No wave filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for wave ID: {self.selected_wave}")

        # Extract the column with wave information.
        wave_data = data[:, self.col_id]
        tolerance = 1e-5  # Tolerance to handle floating-point precision issues.
        mask = torch.abs(wave_data - self.selected_wave) < tolerance
        filtered_data = data[mask]

        return filtered_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """
        Internal transform method.

        This method is used by the underlying transformation framework to apply the transform.

        Args:
            inpt (Any): Input data to be transformed.
            params (dict[str, Any]): Additional parameters (unused).

        Returns:
            Any: The transformed data.
        """
        return self(inpt)
