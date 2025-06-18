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
        self.__save_train_data()
        self.__save_latent_probes()

        results = self.__analyze_results()
        results.validate_results()
        results.process_results()

        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()
        return results

    def __save_reconstruction_data(self) -> None:
        """Compute and save reconstructions and latent representations for test data."""
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

                original_data_list.append(data.cpu().numpy())
                original_covariates_list.append(covariates.cpu().numpy())
                reconstruction_data_list.append(recon_batch.cpu().numpy())
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        original_data = np.concatenate(original_data_list, axis=0)
        original_covariates = np.concatenate(original_covariates_list, axis=0)
        reconstruction_data = np.concatenate(reconstruction_data_list, axis=0)
        z_mean_data = np.concatenate(latent_mean_list, axis=0)
        z_logvar_data = np.concatenate(latent_logvar_list, axis=0)

        skipped_data_df = self.dataloader.get_skipped_data()
        if (
            skipped_data_df is not None
            and skipped_data_df.shape[0] != original_data.shape[0]
        ):
            raise DataRowMismatchError(
                f"Mismatch in skipped data rows ({skipped_data_df.shape[0]}) "
                f"and dataset rows ({original_data.shape[0]})."
            )

        if self.covariate_embedding_technique == "disentangle_embedding":
            uninformed_size = self.properties.model.components.get(
                self.properties.model.architecture
            ).get("latent_dim")
            total_latent_dim = z_mean_data.shape[1]
            sensitive_dim = total_latent_dim - uninformed_size
            z_mean_data = z_mean_data[:, sensitive_dim:]
            z_logvar_data = z_logvar_data[:, sensitive_dim:]

        feature_names = self.dataloader.get_feature_labels()
        covariate_names = self.dataloader.get_encoded_covariate_labels()

        original_col_names = [f"orig_{col}" for col in feature_names]
        original_covariate_names = [f"orig_{col}" for col in covariate_names]
        recon_col_names = [f"recon_{col}" for col in feature_names]
        recon_covariate_names = [f"recon_{col}" for col in covariate_names]
        z_mean_col_names = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logvar_col_names = [f"z_logvar_{i}" for i in range(z_mean_data.shape[1])]

        if self.covariate_embedding_technique in {
            "input_feature_embedding",
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

        df = pd.DataFrame(combined_data, columns=all_columns)

        if skipped_data_df is not None:
            df = pd.concat([skipped_data_df.reset_index(drop=True), df], axis=1)

        output_extension = get_internal_file_extension()
        output_file_path = (
            f"{self.properties.system.output_dir}/reconstructions/"
            f"validation_data.{output_extension}"
        )
        self.logger.info(f"Saving validation data to {output_file_path}...")
        save_data(df, output_file_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="test")

    def __save_train_data(self) -> None:
        """Compute and save reconstructions and latent representations for training data."""
        self.logger.info("Saving latent and reconstruction data per sample.")

        original_data_list = []
        original_covariates_list = []
        reconstruction_data_list = []
        latent_mean_list = []
        latent_logvar_list = []

        with torch.no_grad():
            for batch in tqdm(
                self.train_dataloader, desc="Collecting reconstruction data"
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

                original_data_list.append(data.cpu().numpy())
                original_covariates_list.append(covariates.cpu().numpy())
                reconstruction_data_list.append(recon_batch.cpu().numpy())
                latent_mean_list.append(z_mean.cpu().numpy())
                latent_logvar_list.append(z_logvar.cpu().numpy())

        original_data = np.concatenate(original_data_list, axis=0)
        original_covariates = np.concatenate(original_covariates_list, axis=0)
        reconstruction_data = np.concatenate(reconstruction_data_list, axis=0)
        z_mean_data = np.concatenate(latent_mean_list, axis=0)
        z_logvar_data = np.concatenate(latent_logvar_list, axis=0)

        skipped_data_df = self.dataloader.get_skipped_data(dataloader="train")
        if (
            skipped_data_df is not None
            and skipped_data_df.shape[0] != original_data.shape[0]
        ):
            raise DataRowMismatchError(
                f"Mismatch in skipped data rows ({skipped_data_df.shape[0]}) "
                f"and dataset rows ({original_data.shape[0]})."
            )

        if self.covariate_embedding_technique == "disentangle_embedding":
            uninformed_size = self.properties.model.components.get(
                self.properties.model.architecture
            ).get("latent_dim")
            total_latent_dim = z_mean_data.shape[1]
            sensitive_dim = total_latent_dim - uninformed_size
            z_mean_data = z_mean_data[:, sensitive_dim:]
            z_logvar_data = z_logvar_data[:, sensitive_dim:]

        feature_names = self.dataloader.get_feature_labels()
        covariate_names = self.dataloader.get_encoded_covariate_labels()

        original_col_names = [f"orig_{col}" for col in feature_names]
        original_covariate_names = [f"orig_{col}" for col in covariate_names]
        recon_col_names = [f"recon_{col}" for col in feature_names]
        recon_covariate_names = [f"recon_{col}" for col in covariate_names]
        z_mean_col_names = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logvar_col_names = [f"z_logvar_{i}" for i in range(z_mean_data.shape[1])]

        if self.covariate_embedding_technique in {
            "input_feature_embedding",
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

        df = pd.DataFrame(combined_data, columns=all_columns)

        if skipped_data_df is not None:
            df = pd.concat([skipped_data_df.reset_index(drop=True), df], axis=1)

        output_extension = get_internal_file_extension()
        output_file_path = (
            f"{self.properties.system.output_dir}/reconstructions/"
            f"train_data.{output_extension}"
        )
        self.logger.info(f"Saving train data to {output_file_path}...")
        save_data(df, output_file_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="train")

    def __save_latent_probes(self) -> None:
        """
        Generate reconstructions at fixed latent-space points and save each in its own file.

        For each latent dimension i, two probes at -3 and +3 are decoded. Each probe’s
        reconstruction is saved as a single-row file named:
            latent_space_probes_z{i}_{val}.{ext}
        where columns are the feature names.
        """
        self.logger.info("Generating fixed-point latent-space reconstructions.")
        vae = self.model
        device = self.device

        # Determine dimensions
        cov_dim = vae.embedding_strategy.cov_dim + 1
        latent_plus_cov = vae.decoder_input_dim
        latent_dim = latent_plus_cov - cov_dim

        # Build covariate template (age=0, male one-hot, etc.)
        cov_info = vae.embedding_strategy.covariate_info
        cov_template = torch.zeros((1, cov_dim), device=device)
        print(cov_template)
        cat_groups = cov_info.get("categorical", {})
        print(cat_groups)
        if "sex" in cat_groups:
            sex_idx = cat_groups["sex"]
            one_hot = torch.zeros(len(sex_idx), device=device)
            one_hot[0] = 1.0
            cov_template[0, sex_idx] = one_hot

        # Build latent probes and names
        probes = []
        probe_names = []
        for i in range(latent_dim):
            for val in (-3.0, 3.0):
                z = torch.zeros((1, latent_dim), device=device)
                z[0, i] = val
                probes.append(z)
                probe_names.append(f"z{i}_{int(val)}")

        z_batch = torch.cat(probes, dim=0)
        cov_batch = cov_template.repeat(len(probes), 1)

        # Decode all at once
        with torch.no_grad():
            decoder_input = torch.cat([z_batch, cov_batch], dim=1)
            reconstructions = vae.decoder(decoder_input)  # (2*latent_dim, output_dim)
        recon_np = reconstructions.cpu().numpy()

        # Feature labels
        feature_names = self.dataloader.get_feature_labels()

        # Save each probe to its own file
        ext = get_internal_file_extension()
        base_dir = f"{self.properties.system.output_dir}/reconstructions"
        for probe_name, recon in zip(probe_names, recon_np):
            df = pd.DataFrame(recon.reshape(1, -1), columns=feature_names)
            out_path = f"{base_dir}/latent_space_probes_{probe_name}.{ext}"
            self.logger.info(f"Saving latent probe '{probe_name}' to {out_path}")
            save_data(df, out_path)

        self.logger.info(
            "All fixed-point latent-space reconstructions saved successfully."
        )

    def __analyze_results(self) -> TaskResult:
        """Analyze the validation results using the saved reconstruction and latent data."""
        data_type = self.properties.dataset.data_type
        engine = create_analysis_engine(data_type)
        engine.initialize_engine(
            feature_labels=self.dataloader.get_feature_labels(),
            covariate_labels=self.dataloader.get_encoded_covariate_labels(),
            target_labels=self.dataloader.get_target_labels(),
        )

        results = engine.run_analysis()
        return results

    def __save_latent_parameters(
        self, z_mean_data: np.ndarray, data: str = "train"
    ) -> None:
        """Compute and save the mean and standard deviation for each latent dimension."""
        self.logger.info("Computing learned latent space parameters.")

        latent_means = np.mean(z_mean_data, axis=0)
        latent_stds = np.std(z_mean_data, axis=0)

        latent_df = pd.DataFrame(
            {
                "latent_dim": np.arange(z_mean_data.shape[1]),
                "mean": latent_means,
                "std": latent_stds,
            }
        )

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
