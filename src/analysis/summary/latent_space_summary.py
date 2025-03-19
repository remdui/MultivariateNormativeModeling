"""Module for latent space summary."""

from typing import Any

import pandas as pd


def extract_latent_params(df: pd.DataFrame) -> dict:
    """Extract latent parameters from a DataFrame."""
    params = {}
    for _, row in df.iterrows():
        dim = int(row["latent_dim"])
        params[dim] = [row["mean"], row["std"]]
    return params


def compute_latent_average(df: pd.DataFrame) -> list[float]:
    """Compute the average of means and standard deviations for a DataFrame."""
    mean_of_means = float(df["mean"].mean())
    mean_of_std = float(df["std"].mean())
    return [mean_of_means, mean_of_std]


def summarize_latent_space(engine: Any) -> dict:
    """Summarize latent space parameters and averages."""
    train_latent_params = extract_latent_params(engine.latent_train_df)
    test_latent_params = extract_latent_params(engine.latent_test_df)
    train_average = compute_latent_average(engine.latent_train_df)
    test_average = compute_latent_average(engine.latent_test_df)
    deviation = {}
    mean_diffs = []
    std_diffs = []

    common_dims = sorted(
        set(train_latent_params.keys()) & set(test_latent_params.keys())
    )
    for dim in common_dims:
        train_mean, train_std = train_latent_params[dim]
        test_mean, test_std = test_latent_params[dim]
        mean_diff = train_mean - test_mean
        std_diff = train_std - test_std
        deviation[dim] = {"mean_diff": mean_diff, "std_diff": std_diff}
        mean_diffs.append(abs(mean_diff))
        std_diffs.append(abs(std_diff))

    total_dims = len(common_dims)
    if total_dims == 0:
        average_deviation = {"mean": float("nan"), "std": float("nan")}
    else:
        average_deviation = {
            "mean": sum(mean_diffs) / total_dims,
            "std": sum(std_diffs) / total_dims,
        }

    return {
        "train_latent_params": train_latent_params,
        "test_latent_params": test_latent_params,
        "train_average": train_average,
        "test_average": test_average,
        "deviation": deviation,
        "average_deviation": average_deviation,
    }
