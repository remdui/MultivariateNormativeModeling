"""Module for limiting number of samples."""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SampleLimitTransform(Transform):
    """Transform that randomly keeps a maximum number of samples,.

    setting all other rows to NaN so they can be dropped by subsequent transforms.
    """

    def __init__(self, max_samples: int = 1000, shuffle: bool = True) -> None:
        """Initialize the transform with a limit on how many samples to retain.

        Args:
            max_samples (int): The maximum number of samples to keep.
            shuffle (bool): If True, the rows are randomly permuted before selection.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.max_samples = max_samples
        self.shuffle = shuffle

    def __call__(self, data: Tensor) -> Tensor:
        """Apply the SampleLimit transformation.

        Args:
            data (Tensor): The input data with shape [num_rows, num_features].

        Returns:
            Tensor: A tensor where only up to max_samples rows are kept as-is,
                    and the rest are set to NaN.
        """
        num_rows = data.size(0)
        self.logger.info(
            f"Applying SampleLimitTransform with max {self.max_samples} samples."
        )

        # If the number of rows is within the limit, do nothing
        if num_rows <= self.max_samples:
            self.logger.info(
                "The number of samples is already within the limit. Returning original data."
            )
            return data

        # Shuffle (if requested) and pick the rows we want to keep
        if self.shuffle:
            # Get a random permutation of row indices
            perm = torch.randperm(num_rows, device=data.device)
        else:
            # If not shuffling, just use a straightforward range
            perm = torch.arange(num_rows, device=data.device)

        selected_indices = perm[: self.max_samples]

        # Create a mask of which rows to keep (True) vs. which to set to NaN (False)
        keep_mask = torch.zeros(num_rows, dtype=torch.bool, device=data.device)
        keep_mask[selected_indices] = True

        # Set rows not in keep_mask to NaN
        data[~keep_mask] = float("nan")

        self.logger.info(
            f"Kept {self.max_samples} rows, set {num_rows - self.max_samples} rows to NaN."
        )

        return data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the limit sampling transformation."""
        return self(inpt)
