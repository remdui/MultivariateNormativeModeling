"""Validator class module."""

import numpy as np
import pandas as pd
import torch
from torch import autocast
from tqdm import tqdm

from analysis.engine.factory import create_analysis_engine
from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import (
    get_internal_file_extension,
    save_data,
    write_results_to_file,
)
from util.model_utils import load_model


class ValidateTask(AbstractTask):
    """Validator class to validate the model."""

    def __init__(self) -> None:
        """Initialize the Validator class."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ValidateTask.")
        self.__init_validation_task()

    def __init_validation_task(self) -> None:
        """Setup the validation task."""
        self.task_name = "validate"
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)
        # Get the validation properties
        self.data_representation = self.properties.validation.data_representation
        self.model_file = self.properties.validation.model
        self.model_path = self.properties.system.models_dir + "/" + self.model_file

        # Load model state dictionary from model file
        self.model = load_model(self.model, self.model_path, self.device)

    def run(self) -> TaskResult:
        """
        Main entry point:

        1) Save reconstruction and latent data for each sample.
        2) Analyze the saved data (compute metrics, produce visual samples, etc.).
        """
        self.logger.info("Starting the validation process.")

        # 1) Save reconstructions and test latent distribution
        self.__save_reconstruction_data()

        # 2) Save training latent distribution
        self.__save_training_distribution()

        # 3) Analyze the results using the newly saved data
        results = self.__analyze_results()
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()
        return results

    def __save_training_distribution(self) -> None:
        """Goes through the training set, computes latent representations parameters."""
        self.logger.info("Saving training latent parameters.")

        latent_mean_list = []
        latent_logvar_list = []

        with torch.no_grad():
            for batch in tqdm(
                self.train_dataloader, desc="Calculating latent parameters"
            ):
                data, covariates = (
                    batch  # (inputs, labels) but for unsupervised this may be empty
                )
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    _, z_mean, z_logvar = self.model(data, covariates)

                # Move data to CPU for concatenation
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        # Concatenate all batches into final arrays
        z_mean_data = np.concatenate(latent_mean_list, axis=0)

        self.__save_latent_parameters(z_mean_data, data="train")

    def __save_reconstruction_data(self) -> None:
        """
        Goes through the test set, computes reconstructions and latent representations,.

        and saves them (alongside the originals) to a file in the same order they appear.
        """
        self.logger.info("Saving latent and reconstruction data per sample.")

        original_data_list = []
        reconstruction_data_list = []
        latent_mean_list = []
        latent_logvar_list = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Collecting recon data"):
                data, covariates = (
                    batch  # (inputs, labels) but for unsupervised this may be empty
                )
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    recon_batch, z_mean, z_logvar = self.model(data, covariates)

                # Move data to CPU for concatenation
                original_data_list.append(data.cpu().numpy())
                reconstruction_data_list.append(recon_batch.cpu().numpy())
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        # Concatenate all batches into final arrays
        original_data = np.concatenate(original_data_list, axis=0)
        reconstruction_data = np.concatenate(reconstruction_data_list, axis=0)
        z_mean_data = np.concatenate(latent_mean_list, axis=0)
        z_logvar_data = np.concatenate(latent_logvar_list, axis=0)

        # Retrieve skipped columns data
        skipped_data_df = self.dataloader.get_skipped_data()

        if skipped_data_df is not None:
            if skipped_data_df.shape[0] != original_data.shape[0]:
                raise ValueError(
                    f"Mismatch in skipped data rows ({skipped_data_df.shape[0]}) and dataset rows ({original_data.shape[0]})."
                )

        # Build a single DataFrame that keeps all information.
        # For tabular data, you can merge original & reconstructed columns directly.
        # For image data, you might keep them flattened, or store them differently.
        self.logger.info(
            "Creating DataFrame with original, reconstruction, latent data."
        )

        # Obtain the feature names from your dataloader
        feature_names = self.dataloader.get_feature_labels()
        covariate_names = self.dataloader.get_covariate_labels()

        # Original columns
        original_col_names = [f"orig_{col}" for col in feature_names]

        # Reconstruction columns
        recon_col_names = [f"recon_{col}" for col in feature_names]

        # Latent columns
        z_mean_col_names = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logvar_col_names = [f"z_logvar_{i}" for i in range(z_mean_data.shape[1])]

        # Combine all into a single np.array
        combined_data = np.concatenate(
            [original_data, reconstruction_data, z_mean_data, z_logvar_data], axis=1
        )

        if self.covariate_embedding_technique == "no_embedding":
            all_columns = (
                original_col_names
                + recon_col_names
                + z_mean_col_names
                + z_logvar_col_names
            )
        else:
            all_columns = (
                original_col_names
                + covariate_names
                + recon_col_names
                + z_mean_col_names
                + z_logvar_col_names
            )
        df = pd.DataFrame(combined_data, columns=all_columns)

        # Prepend skipped columns if available
        if skipped_data_df is not None:
            df = pd.concat([skipped_data_df.reset_index(drop=True), df], axis=1)

        # Decide where to save
        output_extension = get_internal_file_extension()
        output_file_path = f"{self.properties.system.output_dir}/reconstructions/validation_data.{output_extension}"

        # Use your custom `save_data` function to handle CSV/HDF format
        self.logger.info(f"Saving validation data to {output_file_path}...")
        save_data(df, output_file_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="test")

    def __analyze_results(self) -> TaskResult:
        """
        Loads or reuses the newly saved reconstruction & latent data,.

        then performs analysis (metrics, visual samples, etc.).
        """

        results = TaskResult()

        # Create analysis engine instance
        data_type = self.properties.dataset.data_type
        engine = create_analysis_engine(data_type)

        # Collect the label labels from dataloader
        feature_labels = self.dataloader.get_feature_labels()
        covariate_labels = self.dataloader.get_covariate_labels()
        target_labels = self.dataloader.get_target_labels()

        # Initialize data analysis engine
        engine.initialize_engine(
            feature_labels=feature_labels,
            covariate_labels=covariate_labels,
            target_labels=target_labels,
        )

        # Calculate model performance and characteristics
        if self.properties.data_analysis.features.reconstruction_mse:
            results["recon_mse"] = engine.calculate_reconstruction_mse()

        if self.properties.data_analysis.features.reconstruction_r2:
            results["recon_r2"] = engine.calculate_reconstruction_r2()

        if self.properties.data_analysis.features.outlier_detection:
            outlier_result_dict = engine.detect_outliers()
            results["outlier_detection"] = outlier_result_dict

        if self.properties.data_analysis.features.latent_space_analysis:
            summary_latent_space_dict = engine.summarize_latent_space()
            results["summary_latent_space"] = summary_latent_space_dict

        latent_outliers = engine.find_extreme_outliers_in_latent(top_k=1)
        results["latent_outliers"] = latent_outliers

        # Generate plots and other visualisations
        if self.properties.data_analysis.features.distribution_plots:
            engine.plot_feature_distributions()

        if self.properties.data_analysis.features.latent_space_visualization:
            engine.plot_latent_distributions(split="train")
            engine.plot_latent_distributions(split="test")
            engine.plot_latent_projection(
                method="pca", n_components=2, color_covariate="Age"
            )
            engine.plot_latent_projection(
                method="pca", n_components=3, color_covariate="Age"
            )
            engine.plot_latent_projection(
                method="tsne", n_components=2, color_covariate="Age"
            )
            engine.plot_latent_projection(
                method="tsne", n_components=3, color_covariate="Age"
            )
            engine.plot_latent_projection(
                method="pca", n_components=2, color_covariate="Sex"
            )
            engine.plot_latent_projection(
                method="pca", n_components=3, color_covariate="Sex"
            )
            engine.plot_latent_projection(
                method="tsne", n_components=2, color_covariate="Sex"
            )
            engine.plot_latent_projection(
                method="tsne", n_components=3, color_covariate="Sex"
            )
            engine.plot_latent_pairplot()
            engine.plot_sampled_latent_distributions(n=5)
            engine.plot_kl_divergence_per_latent_dim()

        return results

    def __save_latent_parameters(
        self, z_mean_data: np.ndarray, data: str = "train"
    ) -> None:
        """
        Compute mean and std across dataset for each latent dim and save it.

        Instead of averaging log-variances, we compute the standard deviation
        directly from the empirical z_mean_data distribution.
        """
        self.logger.info("Computing learned latent space parameters.")

        # Compute mean and standard deviation from z_mean_data
        latent_means = np.mean(z_mean_data, axis=0)
        latent_stds = np.std(z_mean_data, axis=0)  # Corrected: std from z_mean_data

        # Build a DataFrame
        latent_df = pd.DataFrame(
            {
                "latent_dim": np.arange(z_mean_data.shape[1]),
                "mean": latent_means,
                "std": latent_stds,
            }
        )

        # Save latent space parameters
        output_extension = get_internal_file_extension()
        latent_output_path = (
            f"{self.properties.system.output_dir}/model/"
            f"latent_space_{data}.{output_extension}"
        )
        self.logger.info(
            f"Saving {data} latent space parameters to {latent_output_path}..."
        )
        save_data(latent_df, latent_output_path)
        self.logger.info("Latent space parameters saved successfully.")
