"""Latent plots."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore


def plot_latent_distributions(engine: Any, split: str = "train") -> None:
    """Plot distributions of latent values."""
    logger = engine.logger
    if split == "train":
        df = engine.latent_train_df
        title_str = "Train"
    elif split == "test":
        df = engine.latent_test_df
        title_str = "Test"
    else:
        logger.warning(f"Invalid split={split}. Use 'train' or 'test'.")
        return

    if df.empty:
        logger.warning(f"No latent data available for split='{split}'. Skipping plot.")
        return

    dims = df["latent_dim"].to_numpy()
    means = df["mean"].to_numpy()
    stds = df["std"].to_numpy()
    x_min = np.min(means - 3 * stds)
    x_max = np.max(means + 3 * stds)
    x = np.linspace(x_min, x_max, 400)
    cmap = plt.cm.get_cmap("tab10", len(dims))

    plt.figure(figsize=(8, 5))
    for i, dim in enumerate(dims):
        mu = means[i]
        sigma = stds[i]
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
        label = f"Dim {int(dim)} (μ={mu:.2f}, σ={sigma:.2f})"
        plt.plot(x, pdf, label=label, color=cmap(i), linewidth=2)
    plt.title(f"Latent Distributions ({title_str})")
    plt.xlabel("Value")
    plt.ylabel("PDF")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    if engine.properties.data_analysis.plots.show_plots:
        plt.show()
    if engine.properties.data_analysis.plots.save_plots:
        output_folder = os.path.join(
            engine.properties.system.output_dir, "visualizations"
        )
        os.makedirs(output_folder, exist_ok=True)
        plot_filename = f"latent_distributions_{split}.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath)
        logger.info(f"Plot saved to {plot_filepath}")
    plt.close()


def plot_latent_projection(
    engine: Any,
    method: str = "pca",
    n_components: int = 2,
    color_covariate: str | None = None,
) -> None:
    """Project latent values into 2D or 3D space."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. No latent data to project.")
        return

    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not z_mean_cols:
        logger.warning("No z_mean_* columns found in recon_df. Skipping projection.")
        return

    z_mean_array = engine.recon_df[z_mean_cols].to_numpy()
    if n_components not in (2, 3):
        logger.warning(f"Unsupported n_components={n_components}; must be 2 or 3.")
        return

    method = method.lower()
    if method not in ("pca", "tsne"):
        logger.warning(f"Unsupported method='{method}'. Use 'pca' or 'tsne'.")
        return

    # Determine coloring based on the covariate.
    color_array = "steelblue"
    add_colorbar = False
    if color_covariate is not None:
        # First, check for a direct column in test_df.
        if color_covariate in engine.test_df.columns:
            cov_values = engine.test_df[color_covariate].to_numpy()
            if np.issubdtype(cov_values.dtype, np.number):
                color_array = cov_values
                add_colorbar = True
            else:
                logger.warning(
                    f"Covariate '{color_covariate}' is not numeric; using single color."
                )
        else:
            # Look for one-hot encoded columns: search for columns that start with the covariate identifier plus underscore.
            candidate_cols = [
                col
                for col in engine.test_df.columns
                if col.startswith(f"{color_covariate}_")
            ]
            if candidate_cols:
                cov_array = engine.test_df[candidate_cols].to_numpy()
                # Convert one-hot to categorical labels.
                color_array = np.argmax(cov_array, axis=1)
                add_colorbar = True
            else:
                logger.warning(
                    f"Covariate '{color_covariate}' not found in test_df. Using single color."
                )

    # Perform dimensionality reduction.
    if method == "pca":
        reducer = PCA(n_components=n_components)
        try:
            z_transformed = reducer.fit_transform(z_mean_array)
        except ValueError as e:
            logger.error(f"Error performing PCA: {e}")
            return
        method_label = "PCA"
    else:
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
        try:
            z_transformed = reducer.fit_transform(z_mean_array)
        except ValueError as e:
            logger.error(f"Error performing t-SNE: {e}")
            return
        method_label = "t-SNE"

    # Plotting.
    fig = plt.figure(figsize=(8, 8))
    if n_components == 2:
        sc = plt.scatter(
            z_transformed[:, 0],
            z_transformed[:, 1],
            c=color_array,
            cmap="viridis" if add_colorbar else None,
            alpha=0.8,
            s=30,
            marker="o",
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{method_label} Projection (2D) of Latent z_mean")
        if add_colorbar:
            cbar = plt.colorbar(sc)
            cbar.set_label(color_covariate)
    else:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            z_transformed[:, 0],
            z_transformed[:, 1],
            z_transformed[:, 2],
            c=color_array,
            cmap="viridis" if add_colorbar else None,
            alpha=0.8,
            s=30,
            marker="o",
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{method_label} Projection (3D) of Latent z_mean")
        if add_colorbar:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_covariate)
    if engine.properties.data_analysis.plots.show_plots:
        plt.show()
    if engine.properties.data_analysis.plots.save_plots:
        output_folder = os.path.join(
            engine.properties.system.output_dir, "visualizations"
        )
        os.makedirs(output_folder, exist_ok=True)
        plot_filename = f"latent_{method.lower()}_{n_components}d"
        if color_covariate:
            plot_filename += f"_{color_covariate}"
        plot_filename += ".png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        logger.info(
            f"{method_label} ({n_components}D) latent projection plot saved to {plot_filepath}"
        )
    plt.close(fig)


def plot_latent_pairplot(engine: Any) -> None:
    """Create pairplot of latent values."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. Cannot create pairplot.")
        return
    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not z_mean_cols:
        logger.warning("No z_mean_* columns found in recon_df. Skipping pairplot.")
        return
    sub_df = engine.recon_df[z_mean_cols]
    grid = sns.pairplot(sub_df, diag_kind="kde", corner=False)
    grid.fig.suptitle("Pair Plot of z_mean (Latent Space)", y=1.02)
    if engine.properties.data_analysis.plots.show_plots:
        plt.show()
    if engine.properties.data_analysis.plots.save_plots:
        output_folder = os.path.join(
            engine.properties.system.output_dir, "visualizations"
        )
        os.makedirs(output_folder, exist_ok=True)
        plot_filename = "latent_zmean_pairplot.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        grid.fig.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Latent z_mean pairplot saved to {plot_filepath}")
    plt.close(grid.fig)


def plot_sampled_latent_distributions(engine: Any, n: int = 5) -> None:
    """Plot distributions of sampled latent values."""
    logger = engine.logger
    if engine.recon_df.empty or engine.latent_test_df.empty:
        logger.warning("Either recon_df or latent_test_df is empty. Cannot plot.")
        return
    dims = engine.latent_test_df["latent_dim"].to_numpy()
    means = engine.latent_test_df["mean"].to_numpy()
    stds = engine.latent_test_df["std"].to_numpy()
    x_min = np.min(means - 3 * stds)
    x_max = np.max(means + 3 * stds)
    x = np.linspace(x_min, x_max, 400)
    sampled_df = engine.recon_df.sample(n=min(n, len(engine.recon_df)), random_state=42)
    z_mean_cols = [col for col in sampled_df.columns if col.startswith("z_mean_")]
    num_dims = len(dims)
    colors = plt.cm.get_cmap("tab10", num_dims)
    participant_id_col = engine.properties.dataset.unique_identifier_column
    for _, row in sampled_df.iterrows():
        participant_id = row[participant_id_col]
        plt.figure(figsize=(8, 6))
        for i, dim in enumerate(dims):
            mu = means[i]
            sigma = stds[i]
            pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )
            plt.plot(x, pdf, color="black", alpha=0.6, linewidth=1)
            plt.fill_between(x, pdf, color=colors(i), alpha=0.3)
            plt.axvline(
                mu,
                color=colors(i),
                linestyle="solid",
                linewidth=2,
                label=f"Dist. Dim {int(dim)} Mean={mu:.2f}",
            )
        z_means = row[z_mean_cols].to_numpy()
        for dim_idx, value in enumerate(z_means):
            dim_number = int(dims[dim_idx]) if dim_idx < len(dims) else dim_idx
            plt.axvline(
                value,
                color=colors(dim_idx),
                linestyle="dotted",
                linewidth=2,
                label=f"Sampled z Dim {dim_number}, μ={value:.2f}",
            )
        plt.title(f"Latent Distributions - Participant {participant_id}")
        plt.xlabel("Latent Space Value")
        plt.ylabel("Probability Density")
        plt.ylim(bottom=0)
        plt.legend(loc="upper right", fontsize="small", frameon=False)
        plt.tight_layout()
        if engine.properties.data_analysis.plots.show_plots:
            plt.show()
        if engine.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                engine.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)
            plot_filename = f"latent_sampled_participant_{participant_id}.png"
            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            logger.info(
                f"Sampled latent distribution plot for Participant {participant_id} saved to {plot_filepath}"
            )
        plt.close()


def plot_kl_divergence_per_latent_dim(engine: Any) -> None:
    """Plot KL divergence per latent dimension."""
    logger = engine.logger
    if engine.latent_test_df.empty:
        logger.warning("latent_test_df is empty. Cannot compute KL divergence plot.")
        return
    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    z_logvar_cols = [
        col for col in engine.recon_df.columns if col.startswith("z_logvar_")
    ]
    if not z_mean_cols or not z_logvar_cols:
        logger.warning("Missing z_mean or z_logvar columns in recon_df.")
        return
    means = engine.recon_df[z_mean_cols].to_numpy()
    log_vars = engine.recon_df[z_logvar_cols].to_numpy()
    stds = np.exp(0.5 * log_vars)
    kl_per_dim = 0.5 * (stds**2 + means**2 - 1 - np.log(stds**2))
    kl_mean_per_dim = kl_per_dim.mean(axis=0)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(kl_mean_per_dim)), kl_mean_per_dim)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Average KL Divergence")
    plt.title("KL Divergence Per Latent Dimension")
    if engine.properties.data_analysis.plots.show_plots:
        plt.show()
    if engine.properties.data_analysis.plots.save_plots:
        output_folder = os.path.join(
            engine.properties.system.output_dir, "visualizations"
        )
        os.makedirs(output_folder, exist_ok=True)
        plot_filename = "kl_divergence_per_dim.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        logger.info(f"KL divergence plot saved to {plot_filepath}")
    plt.close()
