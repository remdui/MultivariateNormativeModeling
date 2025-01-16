"""Module for filter specific age."""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class AgeFilterTransform(Transform):
    """Transform for filtering data based on age range."""

    def __init__(
        self,
        age_lowerbound: float = 0.0,
        age_upperbound: float = 100.0,
        col_id: int = 1,
    ) -> None:
        """Initialize the age filter transform.

        Args:
            age_lowerbound (float): Minimum age for filtering (inclusive).
            age_upperbound (float): Maximum age for filtering (inclusive).
            col_id (int): Column index for age information in the dataset.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.age_lowerbound = age_lowerbound
        self.age_upperbound = age_upperbound
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """Filter the data by age range.

        Args:
            data (Tensor): Input dataset with age information.

        Returns:
            Tensor: Filtered dataset containing only rows within the age range.
        """
        self.logger.info(
            f"Filtering data for age range: {self.age_lowerbound}-{self.age_upperbound}"
        )

        age_data = data[:, self.col_id]
        mask = (age_data >= self.age_lowerbound) & (age_data <= self.age_upperbound)
        filtered_data = data[mask]

        return filtered_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the age filter transformation."""
        return self(inpt)
