"""Module for filtering data based on wave value.

This module defines a PyTorch transform that filters rows in an input tensor by
comparing a specified column (containing wave information) to a target wave value.
If no target wave is specified (selected_wave == -1), the original data is returned.
"""

import pandas as pd

from entities.log_manager import LogManager


class WaveFilterTransform:
    """
    Transform that filters input data based on a target wave value.

    This transform examines a specific column of the input DataFrame and returns
    only the rows where the wave value matches the target wave.
    If `selected_wave` is -1, no filtering is applied.
    """

    def __init__(self, selected_wave: int = -1, col_name: str = "wave") -> None:
        """
        Initialize the wave filter transform.

        Args:
            selected_wave (int): The target wave value for filtering. Use -1 to disable filtering.
            col_name (str): The index of the column in the input DataFrame that contains wave information.
        """
        self.logger = LogManager.get_logger(__name__)
        self.selected_wave = selected_wave
        self.col_name = col_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the input DataFrame by the selected wave value.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows matching the target wave.
        """
        if self.selected_wave == -1:
            self.logger.info("No wave filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for wave ID: {self.selected_wave}")
        return data[data[self.col_name] == self.selected_wave]
