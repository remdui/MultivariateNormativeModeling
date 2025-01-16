"""Module for filter specific wave."""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class WaveFilterTransform(Transform):
    """Transform for filtering on wave."""

    def __init__(self, selected_wave: int = -1, col_id: int = 1) -> None:
        """Initialize the wave filter transform.

        Args:
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.selected_wave = selected_wave
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """Filter the data by wave.

        Args:
            data (Tensor): Input dataset with wave information.

        Returns:
            Tensor: Filtered dataset containing only rows matching the selected wave.
        """
        if self.selected_wave == -1:
            self.logger.info("No wave filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for wave ID: {self.selected_wave}")

        wave_data = data[:, self.col_id]
        mask = wave_data == self.selected_wave
        filtered_data = data[mask]

        return filtered_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise transformation."""
        return self(inpt)
