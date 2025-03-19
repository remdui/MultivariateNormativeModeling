"""Tensor utilities."""

import pandas as pd
import torch

from entities.log_manager import LogManager


def get_data_as_tensor(
    df: pd.DataFrame, prefix: str, features: list[str]
) -> torch.Tensor | None:
    """
    Given a DataFrame `df`, a prefix (e.g., "orig" or "recon"), and a list of feature names,.

    build a list of columns (e.g., ["orig_feature1", "orig_feature2", ...]), extract their values,
    and return them as a float32 Torch tensor.
    """
    logger = LogManager.get_logger()
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty.")
        return None

    column_names = [f"{prefix}_{feat}" for feat in features]
    missing_cols = [c for c in column_names if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in DataFrame: {missing_cols}")
        return None

    values = df[column_names].values
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return tensor


def get_latent_cols_as_tensor(
    recon_df: pd.DataFrame, prefix: str
) -> torch.Tensor | None:
    """
    Given a DataFrame `recon_df` and a prefix (e.g., "z_mean_" or "z_logvar_"),.

    extract all matching columns, and return them as a float32 Torch tensor.
    """
    logger = LogManager.get_logger()
    if recon_df is None or recon_df.empty:
        logger.warning("recon_df is None or empty - no latent data available.")
        return None

    latent_cols = [c for c in recon_df.columns if c.startswith(prefix)]
    if not latent_cols:
        logger.warning(f"No columns found with prefix '{prefix}'.")
        return None

    latent_values = recon_df[latent_cols].values
    return torch.as_tensor(latent_values, dtype=torch.float32)
