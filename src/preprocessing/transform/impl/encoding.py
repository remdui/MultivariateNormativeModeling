"""Module for data encoding transforms."""

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties
from util.errors import UnsupportedNormalizationMethodError


class EncodingTransform:
    """
    Transform for encoding and normalizing data.

    This transform applies different feature encoding methods based on the configuration.
    """

    train_stats: dict[str, dict[str, dict[str, float]]] = {}

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the encoding transform.

        Args:
            kwargs (dict): Keyword arguments specifying encoding methods per feature.
        """
        self.logger = LogManager.get_logger(__name__)

        # Extract parameters while providing default values
        self.default_method = kwargs.get("default", "min-max")

        # Define encoding methods
        self.method_map = {
            "one_hot_encoding": kwargs.get("one_hot_encoding", []),
            "z-score": kwargs.get("z-score", []),
            "min-max": kwargs.get("min-max", []),
            "raw": kwargs.get("raw", []),
        }

    @staticmethod
    def save_stats_to_file() -> None:
        """Save train statistics (mean, std, min, max) to a JSON file."""
        properties = Properties.get_instance()
        output_dir = os.path.join(properties.system.data_dir, "processed")
        file_path = os.path.join(output_dir, "normalization_stats.json")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(EncodingTransform.train_stats, f, indent=4)
            LogManager.get_logger(__name__).info(
                f"Normalization stats saved to {file_path}"
            )
        except OSError as e:
            LogManager.get_logger(__name__).error(
                f"Failed to save normalization stats: {e}"
            )

    def _get_or_compute_stats(
        self, method: str, feature: str, data: pd.DataFrame
    ) -> dict[str, float]:
        """
        Retrieve stored train statistics for a feature, or compute and store them if missing.

        Args:
            method (str): Normalization method ("z-score" or "min-max").
            feature (str): Feature name.
            data (pd.DataFrame): The dataset.

        Returns:
            Dict[str, float]: Dictionary containing the computed or retrieved stats.
        """
        if method not in EncodingTransform.train_stats:
            EncodingTransform.train_stats[method] = {}

        if feature in EncodingTransform.train_stats[method]:
            stats = EncodingTransform.train_stats[method][feature]
            self.logger.info(
                f"Using stored train stats for {method} normalization of '{feature}': {stats}"
            )
        else:
            # Compute new stats (train set)
            if method == "z-score":
                mean, std = data[feature].mean(), data[feature].std()
                std = np.maximum(std, 1e-8)  # Avoid division by zero
                stats = {"mean": mean, "std": std}
            elif method == "min-max":
                min_val, max_val = data[feature].min(), data[feature].max()
                stats = {"min": min_val, "max": max_val}
            else:
                raise UnsupportedNormalizationMethodError(
                    f"Unknown normalization method: {method}"
                )

            EncodingTransform.train_stats[method][feature] = stats
            self.logger.info(
                f"Computed {method} normalization for '{feature}': {stats}"
            )

        return stats

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature encoding and normalization.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        self.logger.info("Applying EncodingTransform")
        transformed_data = data.copy()
        one_hot_columns: Any = set()  # Track new one-hot encoded columns

        # 1. Apply One-Hot Encoding
        for feature in self.method_map["one_hot_encoding"]:
            if feature in transformed_data.columns:
                self.logger.info(f"Applying one-hot encoding to {feature}")
                dummies = pd.get_dummies(
                    transformed_data[feature], prefix=feature
                ).astype(np.uint8)
                transformed_data.drop(columns=[feature], inplace=True)
                transformed_data = pd.concat([transformed_data, dummies], axis=1)
                one_hot_columns.update(dummies.columns)

        # 2. Apply Normalization Methods (Z-score & Min-Max)
        for method in ("z-score", "min-max"):
            for feature in self.method_map[method]:
                if feature in transformed_data.columns:
                    stats = self._get_or_compute_stats(
                        method, feature, transformed_data
                    )
                    if method == "z-score":
                        transformed_data[feature] = (
                            transformed_data[feature] - stats["mean"]
                        ) / stats["std"]
                    elif method == "min-max":
                        range_values = np.maximum(
                            stats["max"] - stats["min"], 1e-8
                        )  # Avoid division by zero
                        transformed_data[feature] = (
                            transformed_data[feature] - stats["min"]
                        ) / range_values

        # 3. Apply Default Encoding for Remaining Features
        all_explicitly_transformed = (
            set(sum(self.method_map.values(), [])) | one_hot_columns
        )
        remaining_features = (
            set(transformed_data.columns)
            - all_explicitly_transformed
            - set(self.method_map["raw"])
        )

        for feature in remaining_features:
            self.logger.info(
                f"Applying default ({self.default_method}) encoding to {feature}"
            )

            stats = self._get_or_compute_stats(
                self.default_method, feature, transformed_data
            )

            if self.default_method == "z-score":
                transformed_data[feature] = (
                    transformed_data[feature] - stats["mean"]
                ) / stats["std"]
            elif self.default_method == "min-max":
                range_values = np.maximum(
                    stats["max"] - stats["min"], 1e-8
                )  # Avoid division by zero
                transformed_data[feature] = (
                    transformed_data[feature] - stats["min"]
                ) / range_values
            elif self.default_method == "raw":
                self.logger.info(f"Keeping {feature} unchanged")
            else:
                raise UnsupportedNormalizationMethodError(
                    f"Unknown default encoding method: {self.default_method}"
                )

        self.logger.info("EncodingTransform applied successfully")
        return transformed_data
