"""Module for summarizing input and output feature statistics.

This module returns a nested dictionary with detailed statistics for each input feature
(e.g., brain measures in z-scores) and the corresponding output (reconstructed) feature.
For each feature, statistics include count, mean, standard deviation, minimum, 25th percentile,
median, 75th percentile, and maximum.
"""

from typing import Any

import pandas as pd


def summarize_feature(series: pd.Series) -> dict:
    """Compute summary statistics for a given feature (pandas Series)."""
    summary = {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "25%": float(series.quantile(0.25)),
        "median": float(series.median()),
        "75%": float(series.quantile(0.75)),
        "max": float(series.max()),
    }
    return summary


def summarize_input_output_features(engine: Any) -> dict:
    """
    Summarize statistics for each input (orig_*) and output (recon_*) feature.

    Expects that `engine` has:
      - a DataFrame `recon_df` containing columns for both the original and reconstructed data,
        with column names prefixed by "orig_" for inputs and "recon_" for outputs.
      - a list of feature labels in `engine.feature_labels`.

    Returns a nested dictionary structured as:
    {
        "features": {
            "feature1": {
                "input": { ... summary stats ... },
                "output": { ... summary stats ... }
            },
            "feature2": { ... },
            ...
        }
    }
    """
    df = engine.recon_df
    features_summary = {}

    for feature in engine.feature_labels:
        feature_dict = {}

        # Build expected column names for input and output
        input_col = f"orig_{feature}"
        output_col = f"recon_{feature}"

        # Check if columns exist in the DataFrame
        if input_col in df.columns:
            feature_dict["input"] = summarize_feature(df[input_col])
        else:
            feature_dict["input"] = None

        if output_col in df.columns:
            feature_dict["output"] = summarize_feature(df[output_col])
        else:
            feature_dict["output"] = None

        features_summary[feature] = feature_dict

    return {"features": features_summary}
