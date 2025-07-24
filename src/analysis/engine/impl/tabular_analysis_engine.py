"""Tabular implementation of analysis engine."""

import logging
import os
from typing import Any

import pandas as pd

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from entities.log_manager import LogManager
from tasks.task_result import TaskResult
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


class TabularAnalysisEngine(AbstractAnalysisEngine):
    """Implementation of analysis engine for tabular data."""

    def __init__(self) -> None:
        """Initialize the TabularAnalysisEngine instance."""
        super().__init__(LogManager.get_logger(__name__))

        # Define data types
        self.covariate_labels: list[str] = []
        self.feature_labels: list[str] = []
        self.target_labels: list[str] = []

        # Internal references to loaded DataFrames
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.recon_df: pd.DataFrame = pd.DataFrame()
        self.recon_train_df: pd.DataFrame = pd.DataFrame()
        self.latent_test_df: pd.DataFrame = pd.DataFrame()
        self.latent_train_df: pd.DataFrame = pd.DataFrame()

    def initialize_engine(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TabularAnalysisEngine instance."""
        self.logger.info("Initializing Tabular Data Exploration.")

        # Store labels from kwargs
        self.feature_labels = kwargs.get("feature_labels", [])
        self.covariate_labels = kwargs.get("covariate_labels", [])
        self.target_labels = kwargs.get("target_labels", [])

        # Load processed data
        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data

        if not is_data_file(input_data):
            self.logger.error(f"Invalid data file: {input_data}")
            raise ValueError(f"Invalid data file: {input_data}")

        train_output_path = get_processed_file_path(data_dir, input_data, "train")
        test_output_path = get_processed_file_path(data_dir, input_data, "test")

        self.logger.info("Loading processed training/test data...")
        self.train_df = load_data(train_output_path)
        self.test_df = load_data(test_output_path)
        self.logger.info(f"Loaded train_df: {self.train_df.shape}")
        self.logger.info(f"Loaded test_df: {self.test_df.shape}")

        # Load reconstruction data
        output_extension = get_internal_file_extension()
        recon_file_path = os.path.join(
            self.properties.system.output_dir,
            "reconstructions",
            f"validation_data.{output_extension}",
        )
        self.logger.info(f"Loading reconstruction data from {recon_file_path}...")
        self.recon_df = load_data(recon_file_path)
        self.logger.info(f"Loaded recon_df: {self.recon_df.shape}")

        recon_train_file_path = os.path.join(
            self.properties.system.output_dir,
            "reconstructions",
            f"train_data.{output_extension}",
        )
        self.logger.info(
            f"Loading reconstruction train data from {recon_train_file_path}..."
        )
        self.recon_train_df = load_data(recon_train_file_path)
        self.logger.info(f"Loaded recon_train_df: {self.recon_train_df.shape}")

        # Load latent space parameters
        latent_test_file_path = os.path.join(
            self.properties.system.output_dir,
            "model",
            f"latent_space_test.{output_extension}",
        )
        self.logger.info(
            f"Loading latent space parameters from {latent_test_file_path}..."
        )
        self.latent_test_df = load_data(latent_test_file_path)
        self.logger.info(f"Loaded latent_test_df: {self.latent_test_df.shape}")

        latent_train_file_path = os.path.join(
            self.properties.system.output_dir,
            "model",
            f"latent_space_train.{output_extension}",
        )
        self.logger.info(
            f"Loading latent space parameters from {latent_train_file_path}..."
        )
        self.latent_train_df = load_data(latent_train_file_path)
        self.logger.info(f"Loaded latent_train_df: {self.latent_train_df.shape}")

        self.logger.info("Tabular Data Exploration engine initialization complete.")

    def run_analysis(self) -> TaskResult:
        results = TaskResult()

        ###################################################################################
        # Example usage below. Uncomment and adjust calls and functions to project needs  #
        ###################################################################################

        # if self.properties.data_analysis.features.outlier_detection:
        #     results["outlier_detection"] = detect_outliers(self)
        #     results["latent_outliers"] = find_extreme_outliers_in_latent(self, top_k=1)
        #
        # if self.properties.data_analysis.features.latent_space_analysis:
        #     results["latent_space_statistics"] = summarize_latent_space(self)
        #
        # if self.properties.data_analysis.features.summary_statistics:
        #     results["feature_summary_statistics"] = summarize_input_output_features(
        #         self
        #     )
        #
        # results["recon_pearson"] = calculate_reconstruction_pearson(self)
        #
        # if self.properties.data_analysis.features.reconstruction_mse:
        #     results["recon_mse"] = calculate_reconstruction_mse(self)
        #     results["recon_mse_per_feature"] = calculate_reconstruction_mse_per_feature(
        #         self
        #     )
        #
        # if self.properties.data_analysis.features.reconstruction_r2:
        #     results["recon_r2"] = calculate_reconstruction_r2(self)
        #     results["recon_r2_per_feature"] = calculate_reconstruction_r2_per_feature(
        #         self
        #     )
        #
        # results["recon_distribution_KS"] = compare_feature_distributions_ks(self)
        # results["recon_distribution_BC"] = compare_feature_distributions_bhattacharyya(
        #     self
        # )
        #
        # results["normative_kl"] = calculate_latent_kl(self)
        #
        # if self.properties.data_analysis.features.latent_normality_test:
        #     results["normative_shapiro"] = calculate_latent_normality(self)
        #
        # # Invariant analysis for age (continuous)
        # results["invariant_regression_age"] = calculate_latent_regression_error(
        #     self, "age"
        # )
        # results["invariant_nonlinear_regression_age"] = (
        #     calculate_latent_nonlinear_regression_error(self, "age")
        # )
        # results["invariant_adversarial_age"] = calculate_latent_adversarial_performance(
        #     self, "age"
        # )
        # results["invariant_mi_age"] = calculate_latent_mutual_information(self, "age")
        # results["invariant_cca_age"] = calculate_latent_cca_single(self, "age")
        # results["invariant_corr_coef_age"] = calculate_latent_correlation_coefficients(
        #     self, "age"
        # )
        #
        # # Invariant analysis for sex (one-hot encoded)
        # results["invariant_logistic_sex"] = (
        #     calculate_latent_logistic_classification_error(self, "sex")
        # )
        # results["invariant_nonlinear_classification_sex"] = (
        #     calculate_latent_nonlinear_classification_error(self, "sex")
        # )
        # results["invariant_adversarial_sex"] = calculate_latent_adversarial_performance(
        #     self, "sex", task_type="classification"
        # )
        # results["invariant_mi_sex"] = calculate_latent_mutual_information(self, "sex")
        # results["invariant_cca_sex"] = calculate_latent_cca_single(self, "sex")
        # results["invariant_corr_coef_sex"] = calculate_latent_correlation_coefficients(
        #     self, "sex"
        # )
        #
        # # Invariant analysis for site (one-hot encoded)
        # results["invariant_logistic_site"] = (
        #     calculate_latent_logistic_classification_error(self, "site")
        # )
        # results["invariant_nonlinear_classification_site"] = (
        #     calculate_latent_nonlinear_classification_error(self, "site")
        # )
        # results["invariant_adversarial_site"] = (
        #     calculate_latent_adversarial_performance(
        #         self, "site", task_type="classification"
        #     )
        # )
        # results["invariant_mi_site"] = calculate_latent_mutual_information(self, "site")
        # results["invariant_cca_site"] = calculate_latent_cca_single(self, "site")
        # results["invariant_corr_coef_site"] = calculate_latent_correlation_coefficients(
        #     self, "site"
        # )
        #
        # uni_deviation_df = calculate_univariate_deviation_scores(self)
        # maha_deviation_df = calculate_mahalanobis_deviation_scores(self)
        # save_deviation_scores_to_csv(self, uni_deviation_df, maha_deviation_df)
        #
        # if self.properties.data_analysis.features.distribution_plots:
        #     plot_feature_distributions(self)
        #
        # if self.properties.data_analysis.features.latent_space_visualization:
        #     plot_latent_distributions(self, split="train")
        #     plot_latent_distributions(self, split="test")
        #     for method in ("pca", "tsne"):
        #         for n_components in (2, 3):
        #             for color in ("age", "sex", "site"):
        #                 plot_latent_projection(
        #                     self,
        #                     method=method,
        #                     n_components=n_components,
        #                     color_covariate=color,
        #                 )
        #     plot_latent_pairplot(self)
        #     plot_sampled_latent_distributions(self, n=5)
        #     plot_kl_divergence_per_latent_dim(self)

        return results
