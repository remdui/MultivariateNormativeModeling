"""Module for filter specific site."""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SiteFilterTransform(Transform):
    """Transform for filtering on site."""

    def __init__(self, selected_site: int = -1, col_id: int = 1) -> None:
        """Initialize the site filter transform.

        Args:
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.selected_site = selected_site
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """Filter the data by site.

        Args:
            data (Tensor): Input dataset with site information.

        Returns:
            Tensor: Dataset with rows not matching the selected site set to NaN.
        """
        if self.selected_site == -1:
            self.logger.info("No site filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for site ID: {self.selected_site}")

        site_data = data[:, self.col_id]
        tolerance = 1e-5
        mask = torch.abs(site_data - self.selected_site) < tolerance
        data[~mask] = float("nan")

        return data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the site filter transformation."""
        return self(inpt)
