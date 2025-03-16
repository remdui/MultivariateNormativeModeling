"""Module for adding noise to data with various distributions.

This transform adds noise to an input tensor and then clips the result to the original
min-max range of each feature. Supported noise distributions are 'normal' (Gaussian)
and 'uniform'. For reproducibility, the Gaussian noise is generated using a seeded generator.
"""

import numpy as np
import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties
from util.errors import UnsupportedNoiseDistributionError


class NoiseTransform:
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
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.mean = mean
        self.std = std
        self.distribution = distribution

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add noise to the input data and clip the result to the original data range.

        Args:
            data (pd.DataFrame): Input dataframe to which noise will be added.

        Returns:
            pd.DataFrame: Noisy dataframe, with values clipped to the original min and max per feature.

        Raises:
            UnsupportedNoiseDistributionError: If the specified noise distribution is unsupported.
        """
        # Compute per-feature min and max for later clipping.
        data_min = data.min()
        data_max = data.max()

        if self.distribution == "normal":
            self.logger.info(
                f"Adding Gaussian noise sampled from N({self.mean}, {self.std})"
            )

            # Use a fixed seed for reproducibility.
            np.random.seed(self.properties.general.seed)
            noise = np.random.normal(loc=self.mean, scale=self.std, size=data.shape)

            noisy_data = data + noise

        elif self.distribution == "uniform":
            self.logger.info("Adding uniform noise")

            # Generate uniform noise scaled to the range of the input data.
            range_values = data_max - data_min
            noise = (np.random.rand(*data.shape) * range_values) - (range_values / 2)

            noisy_data = data + noise

        else:
            raise UnsupportedNoiseDistributionError(
                f"Unsupported distribution type: {self.distribution}"
            )

        # Clip the noisy data to the original per-feature min and max values.
        noisy_data = np.clip(noisy_data, data_min, data_max)

        return pd.DataFrame(noisy_data, index=data.index, columns=data.columns)
