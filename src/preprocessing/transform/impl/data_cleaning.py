"""Transform for data cleaning."""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class DataCleaningTransform(Transform):
    """Transform for data cleaning."""

    def __init__(self, drop_na: bool = True, remove_duplicates: bool = True) -> None:
        """Initialize the data cleaner.

        Args:
            drop_na (bool): Whether to drop rows with missing values.
            remove_duplicates (bool): Whether to remove duplicate rows.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.drop_na = drop_na
        self.remove_duplicates = remove_duplicates

    def __call__(self, data: Tensor) -> Tensor:
        """Apply data cleaning transformations.

        Args:
            data (Tensor): Input data to be cleaned.

        Returns:
            Tensor: Cleaned data.
        """
        self.logger.info("Cleaning data")
        initial_row_count = data.size(0)
        if self.drop_na:
            na_mask = ~torch.isnan(data).any(dim=1)
            data = data[na_mask]
            dropped_na_count = initial_row_count - data.size(0)
            self.logger.info(f"Dropped {dropped_na_count} rows due to NaNs.")
        if self.remove_duplicates:
            data = torch.unique(data, dim=0)
            dropped_duplicate_count = initial_row_count - data.size(0)
            self.logger.info(f"Dropped {dropped_duplicate_count} duplicate rows.")
        return data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the normalization transformation."""
        return self(inpt)
