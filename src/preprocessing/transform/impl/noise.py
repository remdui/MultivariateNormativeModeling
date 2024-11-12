"""Module for adding noise to data with various distributions."""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties


class NoiseTransform(Transform):
    """Transform for adding noise to data with various distributions."""

    def __init__(
        self, mean: float = 0.0, std: float = 0.05, distribution: str = "normal"
    ) -> None:
        """Initialize the noise transform.

        Args:
            mean (float): Mean of the Gaussian noise to add (used for 'normal' distribution).
            std (float): Standard deviation of the Gaussian noise to add (used for 'normal' distribution).
            distribution (str): Type of noise to add ('normal' or 'uniform').
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.mean = mean
        self.std = std
        self.distribution = distribution

    def __call__(self, data: Tensor) -> Tensor:
        """Add noise to the data based on the specified distribution and clip values to the original min-max range.

        Args:
            data (Tensor): Input data to which noise will be added.

        Returns:
            Tensor: Data with added noise and clipped to original min-max range.
        """
        # Calculate min and max of each feature for clipping
        data_min, _ = data.min(dim=0, keepdim=True)
        data_max, _ = data.max(dim=0, keepdim=True)

        # Add noise based on the specified distribution
        if self.distribution == "normal":
            self.logger.info(
                f"Adding Gaussian noise sampled from N({self.mean},{self.std})"
            )
            noise = torch.normal(
                mean=self.mean,
                std=self.std,
                size=data.size(),
                device=data.device,
                generator=torch.Generator(device=data.device).manual_seed(
                    self.properties.general.seed
                ),
            )
            noisy_data = data + noise

        elif self.distribution == "uniform":
            self.logger.info("Adding uniform noise")
            noise = (
                torch.rand_like(data) * (data_max - data_min)
                - (data_max - data_min) / 2
            )
            noisy_data = data + noise

        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution}")

        # Clip values to the original min-max range of each feature
        noisy_data = torch.clamp(noisy_data, data_min, data_max)
        return noisy_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the noise transformation."""
        return self(inpt)
