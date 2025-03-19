"""Metrics for latent space analysis."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


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


def calculate_latent_regression_error(engine: Any, covariate: str) -> dict[str, float]:
    """Perform latent regression on the latent space and the covariate."""
    logger = engine.logger
    if covariate in engine.recon_df.columns:
        df = engine.recon_df
    elif covariate in engine.test_df.columns:
        unique_id = engine.properties.dataset.unique_identifier_column
        if unique_id in engine.recon_df.columns and unique_id in engine.test_df.columns:
            df = pd.merge(
                engine.recon_df, engine.test_df[[unique_id, covariate]], on=unique_id
            )
        else:
            logger.error(
                "Unique identifier column not found in both recon_df and test_df for covariate regression."
            )
            return {}
    else:
        logger.error(f"Covariate '{covariate}' not found in recon_df or test_df.")
        return {}

    if not pd.api.types.is_numeric_dtype(df[covariate]):
        logger.error(
            f"Covariate '{covariate}' is not numeric. Regression cannot be performed."
        )
        return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error(
            "No latent space columns found with prefix 'z_mean_' for regression."
        )
        return {}

    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    logger.info(
        f"Latent covariate regression for '{covariate}': MSE={mse_val:.4f}, R²={r2_val:.4f}"
    )
    return {"mse": mse_val, "r2": r2_val}
