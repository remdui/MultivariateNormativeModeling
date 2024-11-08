"""Transform for data normalization."""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class NormalizationTransform(Transform):
    """Transform for data normalization."""

    def __init__(self, method: str = "min-max"):
        """Initialize the normalizer.

        Args:
            method (str): Normalization method ('min-max' or 'z-score').
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.method = method

    def __call__(self, data: Tensor) -> Tensor:
        """Normalize the data.

        Args:
            data (pd.DataFrame): Data to be normalized.

        Returns:
            pd.DataFrame: Normalized data.
        """
        self.logger.info(f"Normalizing data using {self.method} method")
        if self.method == "min-max":
            data_min, _ = data.min(dim=0, keepdim=True)
            data_max, _ = data.max(dim=0, keepdim=True)
            range_values = data_max - data_min
            range_values[range_values == 0] = 1e-6  # Avoid division by zero
            normalized_data = (data - data_min) / range_values

        elif self.method == "z-score":
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            data_std[data_std == 0] = 1e-6  # Avoid division by zero
            normalized_data = (data - data_mean) / data_std

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return normalized_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the normalization transformation."""
        return self(inpt)
