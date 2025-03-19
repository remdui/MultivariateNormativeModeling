"""Feature plots."""

import os
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore


def plot_feature_distributions(engine: Any) -> None:
    """Plot distributions of original and reconstructed features."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. Cannot plot histograms.")
        return

    feature_names = [
        col[5:] for col in engine.recon_df.columns if col.startswith("orig_")
    ]
    num_features = len(feature_names)
    num_cols = 4  # Adjust for readability
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()
    plot_type = engine.properties.data_analysis.plots.distribution_plot_type

    for idx, feature in enumerate(feature_names):
        orig_col = f"orig_{feature}"
        recon_col = f"recon_{feature}"
        if orig_col in engine.recon_df.columns and recon_col in engine.recon_df.columns:
            ax = axes[idx]
            if plot_type == "kde":
                sns.kdeplot(
                    engine.recon_df[orig_col], ax=ax, label="Original", color="blue"
                )
                sns.kdeplot(
                    engine.recon_df[recon_col],
                    ax=ax,
                    label="Reconstructed",
                    color="red",
                )
            elif plot_type == "histogram":
                ax.hist(
                    engine.recon_df[orig_col],
                    bins=30,
                    alpha=0.5,
                    label="Original",
                    color="blue",
                )
                ax.hist(
                    engine.recon_df[recon_col],
                    bins=30,
                    alpha=0.5,
                    label="Reconstructed",
                    color="red",
                )
            ax.set_title(feature)
            ax.legend()

    for ax in axes[num_features:]:
        fig.delaxes(ax)

    plt.tight_layout()
    if engine.properties.data_analysis.plots.show_plots:
        plt.show()
    if engine.properties.data_analysis.plots.save_plots:
        output_folder = os.path.join(
            engine.properties.system.output_dir, "visualizations"
        )
        os.makedirs(output_folder, exist_ok=True)
        plot_filename = "feature_distributions.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Reconstruction histograms saved to {plot_filepath}")
    plt.close()
