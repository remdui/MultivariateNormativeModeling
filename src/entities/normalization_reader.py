"""Module for reading and applying inverse normalization."""

import json
import os

import numpy as np

from entities.log_manager import LogManager
from entities.properties import Properties


class NormalizationReader:
    """
    Reads stored normalization statistics and applies inverse transformation to z-score outputs.

    This allows converting model-predicted z-scores back to real-world values.
    """

    def __init__(self) -> None:
        """Initialize the NormalizationReader and load stored normalization statistics."""
        self.logger = LogManager.get_logger(__name__)
        properties = Properties.get_instance()
        self.file_path = os.path.join(
            properties.system.data_dir, "processed", "normalization_stats.json"
        )

        # Load normalization stats from file
        self.normalization_stats = self._load_stats()

    def _load_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Load the normalization stats JSON file."""
        if not os.path.exists(self.file_path):
            raise ValueError(f"Normalization stats file not found: {self.file_path}")

        try:
            with open(self.file_path, encoding="utf-8") as f:
                stats = json.load(f)
            self.logger.info(f"Loaded normalization stats from {self.file_path}")
            return stats
        except Exception as e:
            self.logger.error(f"Failed to load normalization stats: {e}")
            raise ValueError(f"Could not read normalization stats: {e}") from e

    def inverse_transform(
        self, z_scores: np.ndarray, feature_names: list[str]
    ) -> np.ndarray:
        """
        Convert model-predicted z-scores back to real values using stored mean and std.

        Args:
            z_scores (np.ndarray): Array of z-score outputs (shape: [num_samples, num_features]).
            feature_names (list[str]): List of feature names in the same order as columns in z_scores.

        Returns:
            np.ndarray: Array of real-valued predictions.

        Raises:
            KeyError: If a feature name is missing in the normalization stats.
            ValueError: If a feature name is missing from the normalization stats.
        """
        if "z-score" not in self.normalization_stats:
            raise ValueError("Z-score normalization stats not found in JSON file.")

        real_values = np.zeros_like(z_scores)

        for i, feature in enumerate(feature_names):
            if feature not in self.normalization_stats["z-score"]:
                raise KeyError(f"Feature '{feature}' not found in normalization stats.")

            mean = self.normalization_stats["z-score"][feature]["mean"]
            std = self.normalization_stats["z-score"][feature]["std"]

            real_values[:, i] = (z_scores[:, i] * std) + mean

        self.logger.info("Inverse transformation applied to model outputs.")
        return real_values

    def transform_to_z_score(
        self, real_values: np.ndarray, feature_names: list
    ) -> np.ndarray:
        """
        Convert real values to Z-scores using stored mean and std from the training set.

        Args:
            real_values (np.ndarray): The raw feature values to be transformed.
            feature_names (list): The corresponding feature names.

        Returns:
            np.ndarray: Z-score normalized values.
        """
        z_scores = np.zeros_like(real_values, dtype=np.float32)  # Ensure float32 output

        for i, feature in enumerate(feature_names):
            if feature in self.normalization_stats.get("z-score", {}):
                mean = float(
                    self.normalization_stats["z-score"][feature]["mean"]
                )  # Ensure float
                std = float(
                    self.normalization_stats["z-score"][feature]["std"]
                )  # Ensure float
                std = std if std > 1e-8 else 1e-8  # Prevent division by zero

                z_scores[:, i] = (
                    real_values[:, i] - mean
                ) / std  # Ensure float division

                self.logger.info(
                    f"Feature '{feature}' is encoded: mean={mean:.6f}, std={std:.6f}, Z-score={z_scores[:, i]}"
                )
            else:
                self.logger.warning(
                    f"Feature '{feature}' missing from normalization stats. Using raw values."
                )
                z_scores[:, i] = real_values[:, i]  # Leave unnormalized if missing

        return z_scores
