"""Metrics for normative analysis."""

from typing import Any

import numpy as np


def calculate_latent_kl(engine: Any) -> dict:
    """Calculate KL divergence between latent space and prior."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. Cannot compute KL metrics.")
        return {
            "global_mean_kl": float("nan"),
            "kl_std": float("nan"),
            "kl_per_dim": {},
            "dim_deviations": {},
            "max_dim_deviation": float("nan"),
            "average_dim_deviation": float("nan"),
        }

    # Identify latent columns for mean and log variance
    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    z_logvar_cols = [
        col for col in engine.recon_df.columns if col.startswith("z_logvar_")
    ]

    if not z_mean_cols or not z_logvar_cols:
        logger.warning(
            "Missing latent columns (z_mean_*/z_logvar_*) for computing KL metrics."
        )
        return {
            "global_mean_kl": float("nan"),
            "kl_std": float("nan"),
            "kl_per_dim": {},
            "dim_deviations": {},
            "max_dim_deviation": float("nan"),
            "average_dim_deviation": float("nan"),
        }

    # Convert latent columns to numpy arrays
    means = engine.recon_df[z_mean_cols].to_numpy()
    log_vars = engine.recon_df[z_logvar_cols].to_numpy()

    # Compute KL divergence per latent dimension for each sample:
    # KL = 0.5 * (exp(log_var) + mean^2 - 1 - log_var)
    kl_div = 0.5 * (np.exp(log_vars) + means**2 - 1 - log_vars)

    # Global metrics: average and standard deviation over all samples and dimensions.
    global_mean = np.mean(kl_div)
    kl_std = np.std(kl_div)

    # Compute per-dimension average KL divergence (averaged over samples)
    kl_per_dim_values = np.mean(kl_div, axis=0)
    kl_per_dim = {}
    for col, kl_value in zip(z_mean_cols, kl_per_dim_values):
        try:
            dim = int(col.replace("z_mean_", ""))
        except ValueError:
            dim = col  # fallback to using the column name if conversion fails
        kl_per_dim[dim] = kl_value

    # Compute deviations per dimension relative to the overall global mean.
    dim_deviations = {}
    abs_deviations = []
    for dim, avg_kl in kl_per_dim.items():
        deviation = avg_kl - global_mean
        dim_deviations[dim] = deviation
        abs_deviations.append(abs(deviation))

    max_dim_deviation = max(abs_deviations) if abs_deviations else float("nan")
    average_dim_deviation = np.mean(abs_deviations) if abs_deviations else float("nan")

    logger.info(
        f"KL Metrics: global_mean_kl={global_mean:.4f}, kl_std={kl_std:.4f}, kl_per_dim={kl_per_dim}, "
        f"dim_deviations={dim_deviations}, max_dim_deviation={max_dim_deviation:.4f}, "
        f"average_dim_deviation={average_dim_deviation:.4f}"
    )

    return {
        "global_mean_kl": global_mean,
        "kl_std": kl_std,
        "kl_per_dim": kl_per_dim,
        "dim_deviations": dim_deviations,
        "max_dim_deviation": max_dim_deviation,
        "average_dim_deviation": average_dim_deviation,
    }
