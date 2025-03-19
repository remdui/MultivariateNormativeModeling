"""Module for outlier detection."""

from typing import Any

import numpy as np
import torch

from util.tensor_utils import get_data_as_tensor, get_latent_cols_as_tensor


def detect_outliers(engine: Any, standardized: bool = True) -> dict:
    """Detect outliers in input and recon data."""
    logger = engine.logger
    if not standardized:
        logger.info(
            "Outlier detection is only implemented for standardized data. Skipping."
        )
        return {}

    threshold = engine.properties.data_analysis.features.outlier_threshold

    input_tensor = get_data_as_tensor(engine.recon_df, "orig", engine.feature_labels)
    recon_tensor = get_data_as_tensor(engine.recon_df, "recon", engine.feature_labels)
    z_mean_tensor = get_latent_cols_as_tensor(engine.recon_df, "z_mean_")
    z_varlog_tensor = get_latent_cols_as_tensor(engine.recon_df, "z_logvar_")

    if input_tensor is None or recon_tensor is None:
        logger.warning(
            "Could not detect row-level outliers in input/recon due to missing data/tensors."
        )
        row_outlier_input = torch.zeros(0, dtype=torch.bool)
        row_outlier_recon = torch.zeros(0, dtype=torch.bool)
    else:
        row_outlier_input = (input_tensor.abs() >= threshold).any(dim=1)
        row_outlier_recon = (recon_tensor.abs() >= threshold).any(dim=1)

    if z_mean_tensor is not None:
        row_outlier_latent_mean = (z_mean_tensor.abs() >= threshold).any(dim=1)
        num_outliers_latent_mean = int(row_outlier_latent_mean.sum().item())
    else:
        num_outliers_latent_mean = 0

    if z_varlog_tensor is not None:
        row_outlier_latent_varlog = (z_varlog_tensor.abs() >= threshold).any(dim=1)
        num_outliers_latent_varlog = int(row_outlier_latent_varlog.sum().item())
    else:
        num_outliers_latent_varlog = 0

    num_outliers_input = int(row_outlier_input.sum().item())
    num_outliers_recon = int(row_outlier_recon.sum().item())

    if len(row_outlier_input) == len(row_outlier_recon):
        new_outlier_mask = row_outlier_recon & (~row_outlier_input)
        num_new_outliers = int(new_outlier_mask.sum().item())
    else:
        num_new_outliers = 0

    if len(row_outlier_input) == len(row_outlier_recon):
        same_outliers_mask = row_outlier_input & row_outlier_recon
        num_same_outliers = int(same_outliers_mask.sum().item())
    else:
        num_same_outliers = 0

    per_feature_input = {}
    per_feature_recon = {}
    per_feature_latent = {}

    if input_tensor is not None:
        for feature in engine.feature_labels:
            col_name = f"orig_{feature}"
            if col_name in engine.recon_df.columns:
                values = engine.recon_df[col_name].values
                outlier_mask = abs(values) >= threshold
                num_out = np.sum(outlier_mask)
                if num_out > 0:
                    per_feature_input[feature] = int(num_out)

    if recon_tensor is not None:
        for feature in engine.feature_labels:
            col_name = f"recon_{feature}"
            if col_name in engine.recon_df.columns:
                values = engine.recon_df[col_name].values
                outlier_mask = abs(values) >= threshold
                num_out = np.sum(outlier_mask)
                if num_out > 0:
                    per_feature_recon[feature] = int(num_out)

    latent_cols = [
        c
        for c in engine.recon_df.columns
        if c.startswith("z_mean_") or c.startswith("z_logvar_")
    ]
    for col in latent_cols:
        values = engine.recon_df[col].values
        outlier_mask = abs(values) >= threshold
        num_out = np.sum(outlier_mask)
        if num_out > 0:
            per_feature_latent[col] = int(num_out)

    return {
        "input": num_outliers_input,
        "recon": num_outliers_recon,
        "new_outliers": num_new_outliers,
        "same_outliers": num_same_outliers,
        "latent_mean": num_outliers_latent_mean,
        "latent_varlog": num_outliers_latent_varlog,
        "per_feature_input": per_feature_input,
        "per_feature_recon": per_feature_recon,
        "per_feature_latent": per_feature_latent,
    }


def find_extreme_outliers_in_latent(engine: Any, top_k: int = 5) -> dict:
    """Find extreme outliers in latent space."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. No latent data to analyze.")
        return {}

    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not z_mean_cols:
        logger.warning("No z_mean_* columns found in recon_df. Skipping analysis.")
        return {}

    unique_id_col = engine.properties.dataset.unique_identifier_column
    if unique_id_col not in engine.recon_df.columns:
        logger.error(
            f"Unique identifier column '{unique_id_col}' not found in recon_df."
        )
        return {}

    outlier_dict = {}
    for col in z_mean_cols:
        sorted_df = engine.recon_df[[unique_id_col, col]].sort_values(by=col)
        smallest_outliers = sorted_df.iloc[:top_k][unique_id_col].tolist()
        largest_outliers = sorted_df.iloc[-top_k:][unique_id_col].tolist()
        outlier_dict[col] = {
            "largest_outliers": largest_outliers,
            "smallest_outliers": smallest_outliers,
        }
    return outlier_dict
