"""
Module for model validation.

This module defines the ValidateTask class which validates the model by:
    1. Saving reconstructions and latent space representations.
    2. Computing latent space statistics.
    3. Analyzing results via various metrics and visualizations.
"""

import numpy as np
import pandas as pd
import torch
from torch import autocast
from tqdm import tqdm

from analysis.engine.factory import create_analysis_engine
from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.errors import DataRowMismatchError
from util.file_utils import (
    get_internal_file_extension,
    save_data,
    write_results_to_file,
)
from util.model_utils import load_model


class ValidateTask(AbstractTask):
    """
    ValidateTask performs model validation by saving reconstructions, computing latent statistics,.

    and analyzing the results. It supports both tabular and image data.
    """

    def __init__(self) -> None:
        """Initialize the validation task, set up logging, and prepare the experiment."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ValidateTask.")
        self.__init_validation_task()

    def __init_validation_task(self) -> None:
        """
        Set up the validation experiment:

            - Clears and creates a new experiment directory.
            - Loads validation properties and model.
        """
        self.task_name = "validate"
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)

        # Load validation properties from configuration.
        self.data_representation = self.properties.validation.data_representation
        self.model_file = self.properties.validation.model
        self.model_path = self.properties.system.models_dir + "/" + self.model_file

        self.covariate_embedding_technique = self.properties.model.components.get(
            self.properties.model.architecture
        ).get("covariate_embedding")

        # Load model state from file.
        self.model = load_model(self.model, self.model_path, self.device)

    def run(self) -> TaskResult:
        """
        Run the validation process.

        Steps:
            1. Save reconstructions and latent data for test samples.
            2. Compute latent space parameters for training data.
            3. Analyze the results (metrics, visualizations, etc.).

        Returns:
            TaskResult: The results of the validation, including metrics and analyses.
        """
        self.logger.info("Starting the validation process.")

        self.__save_reconstruction_data()
        self.__save_training_distribution()

        results = self.__analyze_results()
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()
        return results

    def __save_training_distribution(self) -> None:
        """
        Compute and save latent representations' statistics (mean and std) over the training set.

        Iterates through the training dataloader, collects z_mean and z_logvar for each batch,
        and saves the computed latent parameters.
        """
        self.logger.info("Saving training latent parameters.")

        latent_mean_list = []
        latent_logvar_list = []

        with torch.no_grad():
            for batch in tqdm(
                self.train_dataloader, desc="Calculating latent parameters"
            ):
                # Unpack batch (inputs and covariates for unsupervised tasks).
                data, covariates = batch
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    model_outputs = self.model(data, covariates)
                    z_mean = model_outputs.get("z_mean", None)
                    z_logvar = model_outputs.get("z_logvar", None)

                # Move outputs to CPU for further processing.
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        # Concatenate outputs from all batches.
        z_mean_data = np.concatenate(latent_mean_list, axis=0)

        self.__save_latent_parameters(z_mean_data, data="train")

    def __save_reconstruction_data(self) -> None:
        """
        Compute and save reconstructions and latent representations for test data.

        Processes the test set to obtain:
            - Original inputs and covariates.
            - Reconstructed outputs.
            - Latent mean and log-variance.
        Results are saved as a DataFrame that may also include skipped columns.
        """
        self.logger.info("Saving latent and reconstruction data per sample.")

        original_data_list = []
        original_covariates_list = []
        reconstruction_data_list = []
        latent_mean_list = []
        latent_logvar_list = []

        with torch.no_grad():
            for batch in tqdm(
                self.test_dataloader, desc="Collecting reconstruction data"
            ):
                data, covariates = batch
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    model_outputs = self.model(data, covariates)
                    recon_batch = model_outputs["x_recon"]
                    z_mean = model_outputs.get("z_mean", None)
                    z_logvar = model_outputs.get("z_logvar", None)

                # Append data converted to CPU arrays.
                original_data_list.append(data.cpu().numpy())
                original_covariates_list.append(covariates.cpu().numpy())
                reconstruction_data_list.append(recon_batch.cpu().numpy())
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        # Concatenate batch data into full arrays.
        original_data = np.concatenate(original_data_list, axis=0)
        original_covariates = np.concatenate(original_covariates_list, axis=0)
        reconstruction_data = np.concatenate(reconstruction_data_list, axis=0)
        z_mean_data = np.concatenate(latent_mean_list, axis=0)
        z_logvar_data = np.concatenate(latent_logvar_list, axis=0)

        # Retrieve any skipped data columns from the dataloader.
        skipped_data_df = self.dataloader.get_skipped_data()
        if skipped_data_df is not None:
            if skipped_data_df.shape[0] != original_data.shape[0]:
                raise DataRowMismatchError(
                    f"Mismatch in skipped data rows ({skipped_data_df.shape[0]}) and dataset rows ({original_data.shape[0]})."
                )

        self.logger.info(
            "Creating DataFrame with original, reconstruction, and latent data."
        )

        # Obtain column labels.
        feature_names = self.dataloader.get_feature_labels()
        covariate_names = self.dataloader.get_covariate_labels()

        # Define column names for original and reconstructed data.
        original_col_names = [f"orig_{col}" for col in feature_names]
        original_covariate_names = [f"orig_{col}" for col in covariate_names]
        recon_col_names = [f"recon_{col}" for col in feature_names]
        recon_covariate_names = [f"recon_{col}" for col in covariate_names]

        # Define column names for latent space representations.
        z_mean_col_names = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logvar_col_names = [f"z_logvar_{i}" for i in range(z_mean_data.shape[1])]

        # Combine arrays and column names based on the covariate embedding technique.
        if self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
        }:
            combined_data = np.concatenate(
                [
                    original_data,
                    original_covariates,
                    reconstruction_data,
                    z_mean_data,
                    z_logvar_data,
                ],
                axis=1,
            )
            all_columns = (
                original_col_names
                + original_covariate_names
                + recon_col_names
                + recon_covariate_names
                + z_mean_col_names
                + z_logvar_col_names
            )
        else:
            combined_data = np.concatenate(
                [original_data, reconstruction_data, z_mean_data, z_logvar_data],
                axis=1,
            )
            all_columns = (
                original_col_names
                + recon_col_names
                + z_mean_col_names
                + z_logvar_col_names
            )

        # Create a DataFrame to hold all information.
        df = pd.DataFrame(combined_data, columns=all_columns)

        # Prepend skipped columns if available.
        if skipped_data_df is not None:
            df = pd.concat([skipped_data_df.reset_index(drop=True), df], axis=1)

        # Determine output file path and extension.
        output_extension = get_internal_file_extension()
        output_file_path = f"{self.properties.system.output_dir}/reconstructions/validation_data.{output_extension}"

        self.logger.info(f"Saving validation data to {output_file_path}...")
        save_data(df, output_file_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="test")

    def __analyze_results(self) -> TaskResult:
        """
        Analyze the validation results using the saved reconstruction and latent data.

        Initializes the analysis engine with feature, covariate, and target labels, computes
        various metrics (e.g., reconstruction MSE, R2, outlier detection) and generates plots.

        Returns:
            TaskResult: An object containing analysis results.
        """
        results = TaskResult()

        # Create and initialize the analysis engine.
        data_type = self.properties.dataset.data_type
        engine = create_analysis_engine(data_type)
        engine.initialize_engine(
            feature_labels=self.dataloader.get_feature_labels(),
            covariate_labels=self.dataloader.get_covariate_labels(),
            target_labels=self.dataloader.get_target_labels(),
        )

        # Compute metrics based on configuration.
        if self.properties.data_analysis.features.reconstruction_mse:
            results["recon_mse"] = engine.calculate_reconstruction_mse()

        if self.properties.data_analysis.features.reconstruction_r2:
            results["recon_r2"] = engine.calculate_reconstruction_r2()

        if self.properties.data_analysis.features.outlier_detection:
            results["outlier_detection"] = engine.detect_outliers()

        if self.properties.data_analysis.features.latent_space_analysis:
            results["summary_latent_space"] = engine.summarize_latent_space()

        # Identify extreme latent outliers.
        results["latent_outliers"] = engine.find_extreme_outliers_in_latent(top_k=1)

        # Generate visualizations if enabled.
        if self.properties.data_analysis.features.distribution_plots:
            engine.plot_feature_distributions()

        if self.properties.data_analysis.features.latent_space_visualization:
            engine.plot_latent_distributions(split="train")
            engine.plot_latent_distributions(split="test")
            # PCA and t-SNE projections with different parameters.
            for method in ("pca", "tsne"):
                for n_components in (2, 3):
                    for color in ("Age", "Sex"):
                        engine.plot_latent_projection(
                            method=method,
                            n_components=n_components,
                            color_covariate=color,
                        )
            engine.plot_latent_pairplot()
            engine.plot_sampled_latent_distributions(n=5)
            engine.plot_kl_divergence_per_latent_dim()

        return results

    def __save_latent_parameters(
        self, z_mean_data: np.ndarray, data: str = "train"
    ) -> None:
        """
        Compute and save the mean and standard deviation for each latent dimension.

        Instead of averaging log-variances, the standard deviation is computed directly from
        the empirical z_mean_data distribution.

        Args:
            z_mean_data (np.ndarray): Array of latent means collected from the dataset.
            data (str): Identifier for the dataset type ('train' or 'test').
        """
        self.logger.info("Computing learned latent space parameters.")

        # Compute statistics for each latent dimension.
        latent_means = np.mean(z_mean_data, axis=0)
        latent_stds = np.std(z_mean_data, axis=0)

        # Create a DataFrame to store latent parameters.
        latent_df = pd.DataFrame(
            {
                "latent_dim": np.arange(z_mean_data.shape[1]),
                "mean": latent_means,
                "std": latent_stds,
            }
        )

        # Determine output file path for latent parameters.
        output_extension = get_internal_file_extension()
        latent_output_path = f"{self.properties.system.output_dir}/model/latent_space_{data}.{output_extension}"
        self.logger.info(
            f"Saving {data} latent space parameters to {latent_output_path}..."
        )
        save_data(latent_df, latent_output_path)
        self.logger.info("Latent space parameters saved successfully.")
