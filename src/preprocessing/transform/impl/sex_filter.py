"""Module for filter specific sex."""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SexFilterTransform(Transform):
    """Transform for filtering data based on sex."""

    def __init__(self, sex: int, col_id: int = 1) -> None:
        """Initialize the sex filter transform.

        Args:
            sex (int): Sex value to filter by (0 or 1).
            col_id (int): Column index for sex information in the dataset.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.sex = sex
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """Filter the data by sex.

        Args:
            data (Tensor): Input dataset with sex information.

        Returns:
            Tensor: Filtered dataset containing only rows matching the specified sex.
        """
        self.logger.info(f"Filtering data for sex: {self.sex}")

        sex_data = data[:, self.col_id]
        mask = sex_data == self.sex
        filtered_data = data[mask]

        return filtered_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the sex filter transformation."""
        return self(inpt)
