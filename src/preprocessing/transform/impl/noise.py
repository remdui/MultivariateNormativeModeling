"""Module for adding noise to data with various distributions.

This transform adds noise to an input tensor and then clips the result to the original
min-max range of each feature. Supported noise distributions are 'normal' (Gaussian)
and 'uniform'. For reproducibility, the Gaussian noise is generated using a seeded generator.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties
from util.errors import UnsupportedNoiseDistributionError


class NoiseTransform(Transform):
    """
    Transform that adds noise to data with a specified distribution and clips the output.

    The transform supports two noise distributions:
      - 'normal': Gaussian noise with a specified mean and standard deviation.
      - 'uniform': Uniform noise scaled to the range of the input data.

    After noise addition, the output is clipped to the original min-max range for each feature.
    """

    def __init__(
        self, mean: float = 0.0, std: float = 0.05, distribution: str = "normal"
    ) -> None:
        """
        Initialize the noise transform.

        Args:
            mean (float): Mean of the Gaussian noise (used if distribution is 'normal').
            std (float): Standard deviation of the Gaussian noise (used if distribution is 'normal').
            distribution (str): Type of noise to add. Supported values: 'normal', 'uniform'.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.mean = mean
        self.std = std
        self.distribution = distribution

    def __call__(self, data: Tensor) -> Tensor:
        """
        Add noise to the input data and clip the result to the original data range.

        Args:
            data (Tensor): Input tensor to which noise will be added.

        Returns:
            Tensor: Noisy data tensor, with values clipped to the original min and max per feature.

        Raises:
            UnsupportedNoiseDistributionError: If the specified noise distribution is unsupported.
        """
        # Compute per-feature min and max for later clipping.
        data_min, _ = data.min(dim=0, keepdim=True)
        data_max, _ = data.max(dim=0, keepdim=True)

        if self.distribution == "normal":
            self.logger.info(
                f"Adding Gaussian noise sampled from N({self.mean}, {self.std})"
            )
            # Use a generator with a fixed seed for reproducibility.
            generator = torch.Generator(device=data.device).manual_seed(
                self.properties.general.seed
            )
            noise = torch.normal(
                mean=self.mean,
                std=self.std,
                size=data.size(),
                device=data.device,
                generator=generator,
            )
            noisy_data = data + noise

        elif self.distribution == "uniform":
            self.logger.info("Adding uniform noise")
            # Generate uniform noise scaled to the range of the input data.
            noise = (
                torch.rand_like(data) * (data_max - data_min)
                - (data_max - data_min) / 2
            )
            noisy_data = data + noise

        else:
            raise UnsupportedNoiseDistributionError(
                f"Unsupported distribution type: {self.distribution}"
            )

        # Clip the noisy data to the original per-feature min and max values.
        noisy_data = torch.clamp(noisy_data, data_min, data_max)
        return noisy_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """
        Internal method to apply the sample limit transformation.

        This method is called by the underlying transformation framework to apply the transform.

        Args:
            inpt (Any): Input data to be normalized.
            params (dict[str, Any]): Additional parameters (unused).

        Returns:
            Any: The normalized data.
        """
        return self(inpt)
