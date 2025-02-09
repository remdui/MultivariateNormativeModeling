"""Data Analysis module to perform data analysis."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from analysis.metrics.mse import compute_mse
from analysis.metrics.r2 import compute_r2_score
from entities.log_manager import LogManager
from util.file_utils import (
    get_internal_file_extension,
    get_processed_file_path,
    is_data_file,
    load_data,
)

# Silent Matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# Silent Pillow
logging.getLogger("PIL").setLevel(logging.WARNING)


def _extract_latent_params(df: pd.DataFrame) -> dict[int, list]:
    """
    Helper to create a dict {dim -> [mean, std]} from a DataFrame.

    that must have columns 'latent_dim', 'mean', 'std'.
    """
    params = {}
    for _, row in df.iterrows():
        dim = int(row["latent_dim"])
        params[dim] = [row["mean"], row["std"]]
    return params


def _compute_latent_average(df: pd.DataFrame) -> list[float]:
    """Return [mean_of_means, mean_of_std] across all rows in a latent DataFrame."""
    mean_of_means = float(df["mean"].mean())
    mean_of_std = float(df["std"].mean())
    return [mean_of_means, mean_of_std]


class TabularAnalysisEngine(AbstractAnalysisEngine):
    """Class to perform data analysis for tabular data."""

    def __init__(self) -> None:
        """Initialize the DataAnalysis object."""
        super().__init__(LogManager.get_logger(__name__))

        # Define data types
        self.covariate_labels: list[str] = []
        self.feature_labels: list[str] = []
        self.target_labels: list[str] = []

        # Internal references to loaded DataFrames
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.recon_df: pd.DataFrame = pd.DataFrame()
        self.latent_test_df: pd.DataFrame = pd.DataFrame()
        self.latent_train_df: pd.DataFrame = pd.DataFrame()

    def initialize_engine(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tabular exploration pipeline."""
        self.logger.info("Initializing Tabular Data Exploration.")

        # Store references for later usage
        self.feature_labels = kwargs.get("feature_labels", [])
        self.covariate_labels = kwargs.get("covariate_labels", [])
        self.target_labels = kwargs.get("target_labels", [])

        # ---------------------------------------------------------
        # 1) Load the processed train/test data
        # ---------------------------------------------------------
        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data

        if not is_data_file(input_data):
            self.logger.error(f"Invalid data file: {input_data}")
            raise ValueError(f"Invalid data file: {input_data}")

        # Processed data file paths
        train_output_path = get_processed_file_path(data_dir, input_data, "train")
        test_output_path = get_processed_file_path(data_dir, input_data, "test")

        self.logger.info("Loading processed training/test data...")
        # Load the processed data
        self.train_df = load_data(train_output_path)  # e.g. DF with columns
        self.test_df = load_data(test_output_path)  # e.g. DF with columns
        self.logger.info(f"Loaded train_df: {self.train_df.shape}")
        self.logger.info(f"Loaded test_df: {self.test_df.shape}")

        # ---------------------------------------------------------
        # 2) Load the new reconstruction data created by ValidateTask
        # ---------------------------------------------------------
        output_extension = get_internal_file_extension()
        recon_file_path = (
            f"{self.properties.system.output_dir}/reconstructions/"
            f"{self.properties.model_name}_validation_data.{output_extension}"
        )

        self.logger.info(f"Loading reconstruction data from {recon_file_path}...")
        self.recon_df = load_data(recon_file_path)
        self.logger.info(f"Loaded recon_df: {self.recon_df.shape}")

        # ---------------------------------------------------------
        # 3) Load the learned latent space parameters
        # ---------------------------------------------------------
        latent_test_file_path = (
            f"{self.properties.system.output_dir}/model/"
            f"{self.properties.model_name}_latent_space_test.{output_extension}"
        )

        self.logger.info(
            f"Loading latent space parameters from {latent_test_file_path}..."
        )
        self.latent_test_df = load_data(latent_test_file_path)
        self.logger.info(f"Loaded latent_test_df: {self.latent_test_df.shape}")

        latent_train_file_path = (
            f"{self.properties.system.output_dir}/model/"
            f"{self.properties.model_name}_latent_space_train.{output_extension}"
        )

        self.logger.info(
            f"Loading latent space parameters from {latent_train_file_path}..."
        )
        self.latent_train_df = load_data(latent_train_file_path)
        self.logger.info(f"Loaded latent_train_df: {self.latent_train_df.shape}")

        self.logger.info("Tabular Data Exploration engine initialization complete.")

    def _get_data_as_tensor(
        self, df: pd.DataFrame, prefix: str, features: list[str]
    ) -> torch.Tensor | None:
        """
        Given a DataFrame `df`, a prefix like 'orig' or 'recon',.

        and a list of feature names, build a list of columns, extract them,
        and return as a float32 Torch tensor.

        e.g., if prefix='orig', columns become ['orig_feature1', 'orig_feature2', ...].
        """
        # Check if DataFrame is valid
        if df is None or df.empty:
            self.logger.warning("DataFrame is None or empty.")
            return None

        # Build the list of columns
        column_names = [f"{prefix}_{feat}" for feat in features]

        # Check for missing columns
        missing_cols = [c for c in column_names if c not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns in DataFrame: {missing_cols}")
            return None

        # Extract and convert to torch tensor
        values = df[column_names].values  # shape: (num_rows, num_features)
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return tensor

    def _get_latent_cols_as_tensor(self, prefix: str) -> torch.Tensor | None:
        """
        Return a float32 tensor with all columns in self.recon_df that start with `prefix`,.

        e.g., 'z_mean_' or 'z_varlog_'.

        If no matching columns exist or recon_df is empty, returns None.
        """
        if self.recon_df is None or self.recon_df.empty:
            self.logger.warning("recon_df is None or empty - no latent data available.")
            return None

        # Gather columns that start with the given prefix
        latent_cols = [c for c in self.recon_df.columns if c.startswith(prefix)]
        if not latent_cols:
            self.logger.warning(f"No columns found with prefix '{prefix}'.")
            return None

        # Convert selected columns to a float32 tensor
        latent_values = self.recon_df[latent_cols].values
        return torch.as_tensor(latent_values, dtype=torch.float32)

    def calculate_reconstruction_mse(self) -> float:
        """
        Compute MSE between original (input) columns and reconstruction (recon) columns.

        for all features at once, then return it as a float.
        """
        input_tensor = self._get_data_as_tensor(
            self.recon_df, "orig", self.feature_labels
        )
        recon_tensor = self._get_data_as_tensor(
            self.recon_df, "recon", self.feature_labels
        )

        if input_tensor is None or recon_tensor is None:
            self.logger.warning("Unable to calculate MSE due to missing data/tensors.")
            return float("nan")

        # Compute MSE via custom function
        mse_tensor = compute_mse(input_tensor, recon_tensor, metric_type="total")
        mse_value = float(mse_tensor)  # Convert to Python float

        self.logger.info(f"Reconstruction MSE (across all features): {mse_value:.4f}")
        return mse_value

    def calculate_reconstruction_r2(self) -> float:
        """
        Compute R^2 between original (input) columns and reconstruction (recon) columns.

        for all features at once, then return it as a float.
        """
        input_tensor = self._get_data_as_tensor(
            self.recon_df, "orig", self.feature_labels
        )
        recon_tensor = self._get_data_as_tensor(
            self.recon_df, "recon", self.feature_labels
        )

        if input_tensor is None or recon_tensor is None:
            self.logger.warning("Unable to calculate R^2 due to missing data/tensors.")
            return float("nan")

        # Compute R^2 via custom function
        r2_tensor = compute_r2_score(input_tensor, recon_tensor, metric_type="total")
        r2_value = float(r2_tensor)

        self.logger.info(f"Reconstruction R^2 (across all features): {r2_value:.4f}")
        return r2_value

    def detect_outliers(self, standardized: bool = True) -> dict[str, Any]:
        """
        Detect outliers among:

        1) original (input) columns ("orig_{feature}")
        2) reconstructed (recon) columns ("recon_{feature}")
        3) latent columns (split into "z_mean_*" and "z_logvar_*")

        If standardized=True, we assume these columns are already in z-score format
        and consider abs(value) >= threshold to be an outlier.

        Returns a dict with:
            {
                "input": <row-level outliers in input>,
                "recon": <row-level outliers in recon>,
                "new_outliers": <outliers in recon but not in input>,
                "same_outliers": <outliers in both input and recon>,
                "latent_mean": <row-level outliers in z_mean_*>,
                "latent_varlog": <row-level outliers in z_logvar_*>,
                "per_feature_input": {feature_name: count_of_outliers, ...},
                "per_feature_recon": {feature_name: count_of_outliers, ...},
                "per_feature_latent": {column_name: count_of_outliers, ...}
            }
        """
        if not standardized:
            self.logger.info(
                "Outlier detection is only implemented for standardized data. Skipping."
            )
            return {}

        # Get the outlier threshold
        threshold = self.properties.data_analysis.features.outlier_threshold

        # ======================
        # 1) Row-level checks
        # ======================
        # Retrieve input/recon tensors
        input_tensor = self._get_data_as_tensor(
            self.recon_df, "orig", self.feature_labels
        )
        recon_tensor = self._get_data_as_tensor(
            self.recon_df, "recon", self.feature_labels
        )

        # Retrieve two separate latent tensors
        z_mean_tensor = self._get_latent_cols_as_tensor("z_mean_")
        z_varlog_tensor = self._get_latent_cols_as_tensor("z_logvar_")

        if input_tensor is None or recon_tensor is None:
            self.logger.warning(
                "Could not detect row-level outliers in input/recon due to missing data/tensors."
            )
            row_outlier_input = torch.zeros(0, dtype=torch.bool)
            row_outlier_recon = torch.zeros(0, dtype=torch.bool)
        else:
            row_outlier_input = (input_tensor.abs() >= threshold).any(dim=1)
            row_outlier_recon = (recon_tensor.abs() >= threshold).any(dim=1)

        # Latent outliers: z_mean
        if z_mean_tensor is not None:
            row_outlier_latent_mean = (z_mean_tensor.abs() >= threshold).any(dim=1)
            outlier_idx_latent_mean = (
                row_outlier_latent_mean.nonzero(as_tuple=True)[0].cpu().numpy()
            )
            self.logger.debug(
                f"Outlier rows in z_mean: {outlier_idx_latent_mean.tolist()}"
            )
            num_outliers_latent_mean = int(row_outlier_latent_mean.sum().item())
        else:
            num_outliers_latent_mean = 0

        # Latent outliers: z_logvar
        if z_varlog_tensor is not None:
            row_outlier_latent_varlog = (z_varlog_tensor.abs() >= threshold).any(dim=1)
            outlier_idx_latent_varlog = (
                row_outlier_latent_varlog.nonzero(as_tuple=True)[0].cpu().numpy()
            )
            self.logger.debug(
                f"Outlier rows in z_varlog: {outlier_idx_latent_varlog.tolist()}"
            )
            num_outliers_latent_varlog = int(row_outlier_latent_varlog.sum().item())
        else:
            num_outliers_latent_varlog = 0

        # Debug logging for input/recon
        outlier_idx_input = row_outlier_input.nonzero(as_tuple=True)[0].cpu().numpy()
        outlier_idx_recon = row_outlier_recon.nonzero(as_tuple=True)[0].cpu().numpy()
        self.logger.debug(f"Outlier rows in input: {outlier_idx_input.tolist()}")
        self.logger.debug(f"Outlier rows in recon: {outlier_idx_recon.tolist()}")

        num_outliers_input = int(row_outlier_input.sum().item())
        num_outliers_recon = int(row_outlier_recon.sum().item())

        # New outliers in recon (not in input)
        if len(row_outlier_input) == len(row_outlier_recon):
            new_outlier_mask = row_outlier_recon & (~row_outlier_input)
            num_new_outliers = int(new_outlier_mask.sum().item())
            new_outlier_idx = new_outlier_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            self.logger.debug(
                f"New outlier rows in recon (not in input): {new_outlier_idx.tolist()}"
            )
        else:
            num_new_outliers = 0

        # Same outliers in both input & recon
        if len(row_outlier_input) == len(row_outlier_recon):
            same_outliers_mask = row_outlier_input & row_outlier_recon
            num_same_outliers = int(same_outliers_mask.sum().item())
            same_outlier_idx = (
                same_outliers_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            )
            self.logger.debug(
                f"Outlier rows in both input and recon: {same_outlier_idx.tolist()}"
            )
        else:
            num_same_outliers = 0

        # ======================
        # 2) Per-feature checks
        # ======================
        # We'll create three dictionaries:
        #   per_feature_input       -> for "orig_{feature}"
        #   per_feature_recon       -> for "recon_{feature}"
        #   per_feature_latent      -> for any z_mean_*/z_logvar_* columns

        per_feature_input: dict[str, int] = {}
        per_feature_recon: dict[str, int] = {}
        per_feature_latent: dict[str, int] = {}

        # ---- (a) Input columns per-feature outliers ----
        if input_tensor is not None:
            for feature in self.feature_labels:
                col_name = f"orig_{feature}"
                if col_name in self.recon_df.columns:
                    values = self.recon_df[col_name].values  # shape: (N,)
                    outlier_mask = abs(values) >= threshold
                    num_out = np.sum(outlier_mask)
                    if num_out > 0:
                        per_feature_input[feature] = int(num_out)

        # ---- (b) Recon columns per-feature outliers ----
        if recon_tensor is not None:
            for feature in self.feature_labels:
                col_name = f"recon_{feature}"
                if col_name in self.recon_df.columns:
                    values = self.recon_df[col_name].values  # shape: (N,)
                    outlier_mask = abs(values) >= threshold
                    num_out = np.sum(outlier_mask)
                    if num_out > 0:
                        per_feature_recon[feature] = int(num_out)

        # ---- (c) Latent columns (z_mean_*, z_logvar_*) ----
        latent_cols = [
            c
            for c in self.recon_df.columns
            if c.startswith("z_mean_") or c.startswith("z_logvar_")
        ]
        for col in latent_cols:
            values = self.recon_df[col].values  # shape: (N,)
            outlier_mask = abs(values) >= threshold
            num_out = np.sum(outlier_mask)
            if num_out > 0:
                # Use the full column name as the key
                per_feature_latent[col] = int(num_out)

        # Combine everything into one dict
        return {
            # Row-level counts
            "input": num_outliers_input,
            "recon": num_outliers_recon,
            "new_outliers": num_new_outliers,
            "same_outliers": num_same_outliers,
            "latent_mean": num_outliers_latent_mean,
            "latent_varlog": num_outliers_latent_varlog,
            # Per-feature dictionaries
            "per_feature_input": per_feature_input,
            "per_feature_recon": per_feature_recon,
            "per_feature_latent": per_feature_latent,
        }

    def summarize_latent_space(self) -> dict:
        """
        Return a dictionary summarizing train and test latent dimension parameters.

        (mean, std), along with averages across all dimensions, per-dimension
        train-vs-test differences, and separate overall difference metrics for mean
        and std.

        Structure:
        {
            "train_latent_params": { dim -> [mean, std] },
            "test_latent_params": { dim -> [mean, std] },
            "train_average": [avg_mean, avg_std],
            "test_average": [avg_mean, avg_std],
            "deviation": {
                dim -> {
                    "mean_diff": float,
                    "std_diff": float
                },
                ...
            },
            "average_deviation": {
                "mean": float,
                "std": float
            }
        }
        """
        train_latent_params = _extract_latent_params(self.latent_train_df)
        test_latent_params = _extract_latent_params(self.latent_test_df)

        train_average = _compute_latent_average(self.latent_train_df)
        test_average = _compute_latent_average(self.latent_test_df)

        deviation = {}
        mean_diffs = []
        std_diffs = []

        # Only compute diffs for dims that appear in both:
        # (Assuming both sets have the same dimensions—otherwise handle missing dims as needed)
        common_dims = sorted(train_latent_params.keys() & test_latent_params.keys())
        for dim in common_dims:
            train_mean, train_std = train_latent_params[dim]
            test_mean, test_std = test_latent_params[dim]

            mean_diff = train_mean - test_mean
            std_diff = train_std - test_std

            deviation[dim] = {
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            }

            mean_diffs.append(abs(mean_diff))
            std_diffs.append(abs(std_diff))

        # Compute overall difference for mean and std separately
        total_dims = len(common_dims)
        if total_dims == 0:
            # Avoid division by zero if no common dims
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

    def plot_latent_distributions(self, split: str = "train") -> None:
        """
        Plot each latent dimension's Normal-like distribution from either.

        self.latent_train_df or self.latent_test_df, depending on `split`.

        Each line corresponds to one latent dimension, with a label that
        includes the dim ID, mean, and std.

        The distribution is plotted using mean = row['mean'] and std = row['std'].
        """
        if split == "train":
            df = self.latent_train_df
            title_str = "Train"
        elif split == "test":
            df = self.latent_test_df
            title_str = "Test"
        else:
            self.logger.warning(f"Invalid split={split}. Use 'train' or 'test'.")
            return

        if df.empty:
            self.logger.warning(
                f"No latent data available for split='{split}'. Skipping plot."
            )
            return

        # Extract arrays of dims, means, and std
        dims = df["latent_dim"].to_numpy()
        means = df["mean"].to_numpy()
        stds = df["std"].to_numpy()

        # Determine a global plotting range (±3 std from each mean)
        x_min = np.min(means - 3 * stds)
        x_max = np.max(means + 3 * stds)
        x = np.linspace(x_min, x_max, 400)  # 400 points for a smooth curve

        # Use a matplotlib colormap to automatically handle colors
        cmap = plt.cm.get_cmap("tab10", len(dims))

        plt.figure(figsize=(8, 5))  # Optional: set a custom figure size
        for i, dim in enumerate(dims):
            mu = means[i]
            sigma = stds[i]

            # Compute the PDF values for a Normal distribution at each point in x
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

        # Show plot if needed
        if self.properties.data_analysis.plots.show_plots:
            plt.show()

        # Save plot if needed
        if self.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                self.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)
            plot_filename = (
                f"{self.properties.model_name}_latent_distributions_{split}.png"
            )
            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath)
            self.logger.info(f"Plot saved to {plot_filepath}")

    def plot_latent_projection(
        self,
        method: str = "pca",
        n_components: int = 2,
        color_covariate: str | None = None,
    ) -> None:
        """
        Perform dimensionality reduction on the z_mean columns in self.recon_df,.

        using PCA or t-SNE, then plot either a 2D or 3D scatter.

        If `color_covariate` is provided, each data point is colored by the
        corresponding column in self.test_df.

        Parameters:
            method: one of ["pca", "tsne"]
            n_components: 2 or 3
            color_covariate: optional name of a covariate column in self.test_df
        """
        # 1) Basic checks
        if self.recon_df.empty:
            self.logger.warning("recon_df is empty. No latent data to project.")
            return

        z_mean_cols = [
            col for col in self.recon_df.columns if col.startswith("z_mean_")
        ]
        if not z_mean_cols:
            self.logger.warning(
                "No z_mean_* columns found in recon_df. Skipping projection."
            )
            return

        # 2) Gather latent vectors
        z_mean_array = self.recon_df[z_mean_cols].to_numpy()

        if n_components not in (2, 3):
            self.logger.warning(
                f"Unsupported n_components={n_components}; must be 2 or 3."
            )
            return

        method = method.lower()
        if method not in ("pca", "tsne"):
            self.logger.warning(f"Unsupported method='{method}'. Use 'pca' or 'tsne'.")
            return

        # 3) (Optional) Gather color array from covariate
        #    If the user provided a column name, attempt to fetch from self.test_df
        color_array: Any = "steelblue"  # default single color
        add_colorbar = False  # we'll add a colorbar only if numeric data

        if color_covariate is not None:
            if color_covariate in self.test_df.columns:
                cov_values = self.test_df[color_covariate].to_numpy()
                # We'll assume numeric for a continuous colormap:
                # If it's categorical, you'd need a discrete colormap approach.
                if np.issubdtype(cov_values.dtype, np.number):
                    color_array = cov_values
                    add_colorbar = True
                else:
                    self.logger.warning(
                        f"Covariate '{color_covariate}' is not numeric; using single color."
                    )
            else:
                self.logger.warning(
                    f"Covariate '{color_covariate}' not found in test_df. Using single color."
                )

        # 4) Create reducer (PCA or t-SNE) and transform
        if method == "pca":
            reducer = PCA(n_components=n_components)
            try:
                z_transformed = reducer.fit_transform(z_mean_array)
            except ValueError as e:
                self.logger.error(f"Error performing PCA: {e}")
                return
            method_label = "PCA"
        else:  # t-SNE
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
            try:
                z_transformed = reducer.fit_transform(z_mean_array)
            except ValueError as e:
                self.logger.error(f"Error performing t-SNE: {e}")
                return
            method_label = "t-SNE"

        # 5) Plot
        fig = plt.figure(figsize=(8, 8))

        # We'll store the scatter artist in sc so we can add colorbars, etc.
        sc = None

        if n_components == 2:
            # 2D scatter
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

            # If using a numeric covariate, add a colorbar
            if add_colorbar:
                cbar = plt.colorbar(sc)
                cbar.set_label(color_covariate)

        else:  # 3D scatter
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

            # For a 3D plot, we can add a colorbar to the figure if numeric
            if add_colorbar and hasattr(fig, "colorbar"):
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(color_covariate)

        # 6) Show or save
        if self.properties.data_analysis.plots.show_plots:
            plt.show()

        if self.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                self.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)

            plot_filename = (
                f"{self.properties.model_name}_latent_{method.lower()}_{n_components}d"
            )
            if color_covariate:
                plot_filename += f"_{color_covariate}"
            plot_filename += ".png"

            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            self.logger.info(
                f"{method_label} ({n_components}D) latent projection plot saved to {plot_filepath}"
            )
        else:
            plt.close(fig)

    def plot_latent_pairplot(self) -> None:
        """
        Creates a scatterplot matrix (pair plot) of all z_mean columns.

        from recon_df. This helps to visually check for approximate normality
        and relationships across latent dimensions.
        """
        if self.recon_df.empty:
            self.logger.warning("recon_df is empty. Cannot create pairplot.")
            return

        # 1) Select only the columns that start with "z_mean_"
        z_mean_cols = [
            col for col in self.recon_df.columns if col.startswith("z_mean_")
        ]
        if not z_mean_cols:
            self.logger.warning(
                "No z_mean_* columns found in recon_df. Skipping pairplot."
            )
            return

        sub_df = self.recon_df[z_mean_cols]

        # 2) Create the pair plot with Seaborn
        grid = sns.pairplot(sub_df, diag_kind="kde", corner=False)
        grid.fig.suptitle("Pair Plot of z_mean (Latent Space)", y=1.02)

        # 3) Show or save
        if self.properties.data_analysis.plots.show_plots:
            plt.show()

        if self.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                self.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)

            plot_filename = f"{self.properties.model_name}_latent_zmean_pairplot.png"
            plot_filepath = os.path.join(output_folder, plot_filename)
            grid.fig.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"Latent z_mean pairplot saved to {plot_filepath}")
        else:
            plt.close(grid.fig)

    def find_extreme_outliers_in_latent(
        self, top_k: int = 5
    ) -> dict[str, dict[str, list[Any]]]:
        """
        Identify the samples with the largest and smallest outliers in each z_mean_* column.

        Parameters:
            top_k (int): Number of top/bottom extreme outliers to select per z_mean column.

        Returns:
            A dictionary containing:
            {
                "z_mean_<dim>": {
                    "largest_outliers": [id1, id2, ...],
                    "smallest_outliers": [id1, id2, ...]
                },
                ...
            }
        """
        if self.recon_df.empty:
            self.logger.warning("recon_df is empty. No latent data to analyze.")
            return {}

        # Identify z_mean columns
        z_mean_cols = [
            col for col in self.recon_df.columns if col.startswith("z_mean_")
        ]
        if not z_mean_cols:
            self.logger.warning(
                "No z_mean_* columns found in recon_df. Skipping analysis."
            )
            return {}

        # Get the unique identifier column
        unique_id_col = self.properties.dataset.unique_identifier_column
        if unique_id_col not in self.recon_df.columns:
            self.logger.error(
                f"Unique identifier column '{unique_id_col}' not found in recon_df."
            )
            return {}

        outlier_dict = {}

        for col in z_mean_cols:
            # Sort dataframe based on column values
            sorted_df = self.recon_df[[unique_id_col, col]].sort_values(by=col)

            # Select top_k smallest and largest
            smallest_outliers = sorted_df.iloc[:top_k][unique_id_col].tolist()
            largest_outliers = sorted_df.iloc[-top_k:][unique_id_col].tolist()

            outlier_dict[col] = {
                "largest_outliers": largest_outliers,
                "smallest_outliers": smallest_outliers,
            }

        return outlier_dict

    def plot_sampled_latent_distributions(self, n: int = 5) -> None:
        """
        Sample "n" random participants from recon_df and plot their z_mean values.

        against the latent distribution of the test set, with separate plots for each participant.
        Each latent dimension has a unique color. Mean values are plotted as solid lines,
        while participant-specific values are plotted as dotted lines.
        The legend explicitly distinguishes between the distribution mean and sampled values.
        """
        if self.recon_df.empty or self.latent_test_df.empty:
            self.logger.warning(
                "Either recon_df or latent_test_df is empty. Cannot plot."
            )
            return

        # Extract latent test data distributions
        dims = self.latent_test_df["latent_dim"].to_numpy()
        means = self.latent_test_df["mean"].to_numpy()
        stds = self.latent_test_df["std"].to_numpy()

        # Define plotting range
        x_min = np.min(means - 3 * stds)
        x_max = np.max(means + 3 * stds)
        x = np.linspace(x_min, x_max, 400)

        # Sample "n" random participants from recon_df
        sampled_df = self.recon_df.sample(n=min(n, len(self.recon_df)), random_state=42)
        z_mean_cols = [col for col in sampled_df.columns if col.startswith("z_mean_")]

        # Generate unique colors for each latent dimension
        num_dims = len(dims)
        colors = plt.cm.get_cmap("tab10", num_dims)

        participant_id_col = self.properties.dataset.unique_identifier_column

        for _, row in sampled_df.iterrows():
            participant_id = row[participant_id_col]
            plt.figure(figsize=(8, 6))

            # Plot distributions with higher opacity and solid mean lines
            for i, dim in enumerate(dims):
                mu = means[i]
                sigma = stds[i]
                pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x - mu) / sigma) ** 2
                )
                plt.plot(
                    x, pdf, color="black", alpha=0.6, linewidth=1
                )  # Keep density function in black
                plt.fill_between(x, pdf, color=colors(i), alpha=0.3)
                plt.axvline(
                    mu,
                    color=colors(i),
                    linestyle="solid",
                    linewidth=2,
                    label=f"Dist. Dim {int(dim)} Mean={mu:.2f}",
                )

            # Plot participant's z_mean values as dotted vertical lines
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
            plt.ylim(bottom=0)  # Ensure y-axis starts at 0
            plt.legend(loc="upper right", fontsize="small", frameon=False)
            plt.tight_layout()

            # Show or save plot
            if self.properties.data_analysis.plots.show_plots:
                plt.show()

            if self.properties.data_analysis.plots.save_plots:
                output_folder = os.path.join(
                    self.properties.system.output_dir, "visualizations"
                )
                os.makedirs(output_folder, exist_ok=True)
                plot_filename = f"{self.properties.model_name}_latent_sampled_participant_{participant_id}.png"
                plot_filepath = os.path.join(output_folder, plot_filename)
                plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
                self.logger.info(
                    f"Sampled latent distribution plot for Participant {participant_id} saved to {plot_filepath}"
                )
            else:
                plt.close()

    def plot_kl_divergence_per_latent_dim(self) -> None:
        """
        Plot the KL divergence per latent dimension using the test set latent variables.

        This helps visualize which latent dimensions contribute the most to the KL term.
        """
        if self.latent_test_df.empty:
            self.logger.warning(
                "latent_test_df is empty. Cannot compute KL divergence plot."
            )
            return

        # Extract mean and log variance columns from the latent test set
        z_mean_cols = [
            col for col in self.recon_df.columns if col.startswith("z_mean_")
        ]
        z_logvar_cols = [
            col for col in self.recon_df.columns if col.startswith("z_logvar_")
        ]

        if not z_mean_cols or not z_logvar_cols:
            self.logger.warning("Missing z_mean or z_logvar columns in recon_df.")
            return

        # Convert DataFrame columns to numpy arrays
        means = self.recon_df[z_mean_cols].to_numpy()
        log_vars = self.recon_df[z_logvar_cols].to_numpy()
        stds = np.exp(0.5 * log_vars)  # Convert log variance to standard deviation

        # Compute KL divergence per latent dimension
        kl_per_dim = 0.5 * (stds**2 + means**2 - 1 - np.log(stds**2))
        kl_mean_per_dim = kl_per_dim.mean(axis=0)  # Average over all samples

        # Plot KL divergence per latent dimension
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(kl_mean_per_dim)), kl_mean_per_dim)
        plt.xlabel("Latent Dimension")
        plt.ylabel("Average KL Divergence")
        plt.title("KL Divergence Per Latent Dimension")

        # Show or save plot
        if self.properties.data_analysis.plots.show_plots:
            plt.show()

        # Save plot if needed
        if self.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                self.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)
            plot_filename = f"{self.properties.model_name}_kl_divergence_per_dim.png"
            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"KL divergence plot saved to {plot_filepath}")

    def plot_feature_distributions(self) -> None:
        """Plot histograms or density plots comparing original (`orig_*`) and reconstructed (`recon_*`) features."""
        if self.recon_df.empty:
            self.logger.warning("recon_df is empty. Cannot plot histograms.")
            return

        # Identify feature columns
        feature_names = [
            col[5:] for col in self.recon_df.columns if col.startswith("orig_")
        ]
        num_features = len(feature_names)
        num_cols = 4  # Adjust for readability
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        axes = axes.flatten()

        plot_type = self.properties.data_analysis.plots.distribution_plot_type

        for idx, feature in enumerate(feature_names):
            orig_col = f"orig_{feature}"
            recon_col = f"recon_{feature}"

            if orig_col in self.recon_df.columns and recon_col in self.recon_df.columns:
                ax = axes[idx]
                if plot_type == "kde":
                    sns.kdeplot(
                        self.recon_df[orig_col], ax=ax, label="Original", color="blue"
                    )
                    sns.kdeplot(
                        self.recon_df[recon_col],
                        ax=ax,
                        label="Reconstructed",
                        color="red",
                    )
                elif plot_type == "histogram":
                    ax.hist(
                        self.recon_df[orig_col],
                        bins=30,
                        alpha=0.5,
                        label="Original",
                        color="blue",
                    )
                    ax.hist(
                        self.recon_df[recon_col],
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
        if self.properties.data_analysis.plots.show_plots:
            plt.show()
        if self.properties.data_analysis.plots.save_plots:
            output_folder = os.path.join(
                self.properties.system.output_dir, "visualizations"
            )
            os.makedirs(output_folder, exist_ok=True)
            plot_filename = f"{self.properties.model_name}_feature_distributions.png"
            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"Reconstruction histograms saved to {plot_filepath}")
