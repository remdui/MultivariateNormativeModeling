"""NumPy utilities."""

import numpy as np
import pandas as pd

from entities.log_manager import LogManager


def get_data_as_numpy(
    df: pd.DataFrame, prefix: str, features: list[str]
) -> np.ndarray | None:
    """Extracts columns from the DataFrame with the given prefix and feature names converts the data to float32 NumPy array, and returns it."""
    logger = LogManager.get_logger()
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty.")
        return None

    column_names = [f"{prefix}_{feat}" for feat in features]
    missing_cols = [c for c in column_names if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in DataFrame: {missing_cols}")
        return None

    # Directly convert the DataFrame values to a float32 NumPy array
    return df[column_names].values.astype(np.float32)


def get_latent_cols_as_numpy(df: pd.DataFrame, prefix: str) -> np.ndarray | None:
    """Given a DataFrame `df` and a prefix (e.g., "z_mean_" or "z_logvar_") extracts all matching columns, and returns their values as a float32 NumPy array."""
    logger = LogManager.get_logger()
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty - no latent data available.")
        return None

    latent_cols = [c for c in df.columns if c.startswith(prefix)]
    if not latent_cols:
        logger.warning(f"No columns found with prefix '{prefix}'.")
        return None

    return df[latent_cols].values.astype(np.float32)
