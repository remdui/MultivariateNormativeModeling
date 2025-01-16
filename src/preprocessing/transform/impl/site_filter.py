"""Module for filter specific sites."""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SiteFilterTransform(Transform):
    """Transform for filtering on site."""

    def __init__(self, selected_sites: int = -1, site_col_id: int = 1) -> None:
        """Initialize the site filter transform.

        Args:
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.selected_sites = selected_sites
        self.site_col_id = site_col_id

    def __call__(self, data: Tensor) -> Tensor:
        """Filter the data by site.

        Args:
            data (Tensor): Input dataset with site information.

        Returns:
            Tensor: Filtered dataset containing only rows matching the selected site.
        """
        if self.selected_sites == -1:
            self.logger.info("No site filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for site ID: {self.selected_sites}")

        site_data = data[:, self.site_col_id]
        mask = site_data == self.selected_sites
        filtered_data = data[mask]

        return filtered_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise transformation."""
        return self(inpt)
