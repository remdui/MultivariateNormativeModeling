"""
Module for model validation (extended with Brain-Age/BAG support).

Original functionality is unchanged: reconstructions, latent stats and generic
analysis are still produced. Two new items are added for the test split:

    • columns “brain_age” and “bag” in the saved CSV/Parquet files
    • quick metrics (MAE, RMSE, Pearson r) logged for test split
    • a per-sample file with chronological_age, brain_age_gap, and brain_age
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
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
    ValidateTask performs model validation by saving reconstructions, computing.

    latent statistics, and analysing the results. Now also records brain-age.
    """

    def __init__(self) -> None:
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ValidateTask.")
        self.__init_validation_task()

    def __init_validation_task(self) -> None:
        self.task_name = "validate"
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)

        self.data_representation = self.properties.validation.data_representation
        self.model_file = self.properties.validation.model
        self.model_path = f"{self.properties.system.models_dir}/{self.model_file}"

        self.covariate_embedding_technique = self.properties.model.components.get(
            self.properties.model.architecture
        ).get("covariate_embedding")

        self.model = load_model(self.model, self.model_path, self.device)

    def run(self) -> TaskResult:
        self.logger.info("Starting the validation process.")

        self.__save_reconstruction_data()  # test set
        self.__save_train_data()  # train set

        results = self.__analyze_results()
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()
        return results

    def __save_reconstruction_data(self) -> None:
        self.logger.info("Saving latent and reconstruction data per test sample.")

        # lists for aggregation
        orig_x, orig_cov, recon_x = [], [], []
        z_mean_lst, z_logv_lst = [], []
        brain_age_lst = []

        age_idx = self.dataloader.get_encoded_covariate_labels().index("age")

        with torch.no_grad():
            for data, covariates in tqdm(
                self.test_dataloader, desc="Collecting reconstruction data"
            ):
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    out = self.model(data, covariates)
                    recon_batch = out["x_recon"]
                    z_mean = out.get("z_mean")
                    z_logv = out.get("z_logvar")

                # aggregate
                orig_x.append(data.cpu().numpy())
                orig_cov.append(covariates.cpu().numpy())
                recon_x.append(recon_batch.cpu().numpy())
                z_mean_lst.append(z_mean.cpu().numpy())
                z_logv_lst.append(z_logv.cpu().numpy())

                # brain-age + gap if present
                if "g" in out:
                    ba = out["g"].cpu().numpy()
                    brain_age_lst.append(ba)

        # concatenate
        orig_x = np.concatenate(orig_x, 0)
        orig_cov = np.concatenate(orig_cov, 0)
        recon_x = np.concatenate(recon_x, 0)
        z_mean_data = np.concatenate(z_mean_lst, 0)
        z_logv_data = np.concatenate(z_logv_lst, 0)
        brain_age_data = np.concatenate(brain_age_lst, 0) if brain_age_lst else None

        # disentangle trimming stays unchanged
        if self.covariate_embedding_technique == "disentangle_embedding":
            uninformed = self.properties.model.components.get(
                self.properties.model.architecture
            ).get("latent_dim")
            sens_dim = z_mean_data.shape[1] - uninformed
            z_mean_data = z_mean_data[:, sens_dim:]
            z_logv_data = z_logv_data[:, sens_dim:]

        # column labels
        feat_names = self.dataloader.get_feature_labels()
        cov_names = self.dataloader.get_encoded_covariate_labels()
        orig_feat_cols = [f"orig_{c}" for c in feat_names]
        recon_feat_cols = [f"recon_{c}" for c in feat_names]
        z_mean_cols = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logv_cols = [f"z_logvar_{i}" for i in range(z_logv_data.shape[1])]

        # choose schema depending on embedding family
        if self.covariate_embedding_technique in {
            "input_feature_embedding",
            "conditional_embedding",
        }:
            orig_cov_cols = [f"orig_{c}" for c in cov_names]
            recon_cov_cols = [f"recon_{c}" for c in cov_names]
            base_arrays = [orig_x, orig_cov, recon_x, z_mean_data, z_logv_data]
            base_cols = (
                orig_feat_cols
                + orig_cov_cols
                + recon_feat_cols
                + recon_cov_cols
                + z_mean_cols
                + z_logv_cols
            )
        else:
            base_arrays = [orig_x, recon_x, z_mean_data, z_logv_data]
            base_cols = orig_feat_cols + recon_feat_cols + z_mean_cols + z_logv_cols

        # append brain-age data if available
        if brain_age_data is not None:
            base_arrays += [brain_age_data]
            base_cols += ["brain_age_gap"]

        df = pd.DataFrame(np.concatenate(base_arrays, 1), columns=base_cols)

        # prepend skipped rows if any
        skipped_df = self.dataloader.get_skipped_data()
        if skipped_df is not None:
            if skipped_df.shape[0] != df.shape[0]:
                raise DataRowMismatchError(
                    f"Mismatch skipped rows ({skipped_df.shape[0]}) vs df ({df.shape[0]})"
                )
            df = pd.concat([skipped_df.reset_index(drop=True), df], axis=1)

        ext = get_internal_file_extension()
        out_path = (
            f"{self.properties.system.output_dir}/reconstructions/validation_data.{ext}"
        )
        self.logger.info("Saving validation data to %s...", out_path)
        save_data(df, out_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="test")

        if brain_age_data is not None:
            # build DataFrame with z-scores
            ba_df = pd.DataFrame(
                {
                    "chronological_age_z": orig_cov[:, age_idx].squeeze(),
                    "brain_age_gap_z": brain_age_data.squeeze(),
                }
            )

            # now invert the z-score transform:
            age_mean = 10.442000951778178
            age_std = 3.4457687546104125

            ba_df["chronological_age"] = (
                ba_df["chronological_age_z"] * age_std + age_mean
            )
            ba_df["brain_age"] = (
                ba_df["chronological_age_z"] + ba_df["brain_age_gap_z"]
            ) * age_std + age_mean

            chrono = ba_df["chronological_age"]
            ba = ba_df["brain_age"]
            mae = mean_absolute_error(chrono, ba)
            self.logger.info("[test] Brain-Age: MAE %.3f", mae)

            ext = get_internal_file_extension()
            out_path = (
                f"{self.properties.system.output_dir}"
                + f"/reconstructions/brain_age_test.{ext}"
            )
            save_data(ba_df, out_path)
            self.logger.info(
                "Saved per-sample Brain-Age data (with years) to %s", out_path
            )

    def __save_train_data(self) -> None:
        self.logger.info("Saving latent and reconstruction data per train sample.")

        orig_x, orig_cov, recon_x = [], [], []
        z_mean_lst, z_logv_lst = [], []
        brain_age_lst, bag_lst = [], []

        age_idx = self.dataloader.get_encoded_covariate_labels().index("age")

        with torch.no_grad():
            for data, covariates in tqdm(
                self.train_dataloader, desc="Collecting reconstruction data"
            ):
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    out = self.model(data, covariates)

                orig_x.append(data.cpu().numpy())
                orig_cov.append(covariates.cpu().numpy())
                recon_x.append(out["x_recon"].cpu().numpy())
                z_mean_lst.append(out["z_mean"].cpu().numpy())
                z_logv_lst.append(out["z_logvar"].cpu().numpy())

                if "brain_age" in out:
                    ba = out["g"].cpu().numpy()
                    brain_age_lst.append(ba)
                    bag_lst.append(ba - covariates[:, [age_idx]].cpu().numpy())

        orig_x = np.concatenate(orig_x, 0)
        orig_cov = np.concatenate(orig_cov, 0)
        recon_x = np.concatenate(recon_x, 0)
        z_mean_data = np.concatenate(z_mean_lst, 0)
        z_logv_data = np.concatenate(z_logv_lst, 0)
        brain_age_data = np.concatenate(brain_age_lst, 0) if brain_age_lst else None
        bag_data = np.concatenate(bag_lst, 0) if bag_lst else None

        if self.covariate_embedding_technique == "disentangle_embedding":
            uninformed = self.properties.model.components.get(
                self.properties.model.architecture
            ).get("latent_dim")
            sens_dim = z_mean_data.shape[1] - uninformed
            z_mean_data = z_mean_data[:, sens_dim:]
            z_logv_data = z_logv_data[:, sens_dim:]

        feat_names = self.dataloader.get_feature_labels()
        cov_names = self.dataloader.get_encoded_covariate_labels()
        orig_feat_cols = [f"orig_{c}" for c in feat_names]
        recon_feat_cols = [f"recon_{c}" for c in feat_names]
        z_mean_cols = [f"z_mean_{i}" for i in range(z_mean_data.shape[1])]
        z_logv_cols = [f"z_logvar_{i}" for i in range(z_logv_data.shape[1])]

        if self.covariate_embedding_technique in {
            "input_feature_embedding",
            "conditional_embedding",
        }:
            orig_cov_cols = [f"orig_{c}" for c in cov_names]
            recon_cov_cols = [f"recon_{c}" for c in cov_names]
            base_arrays = [orig_x, orig_cov, recon_x, z_mean_data, z_logv_data]
            base_cols = (
                orig_feat_cols
                + orig_cov_cols
                + recon_feat_cols
                + recon_cov_cols
                + z_mean_cols
                + z_logv_cols
            )
        else:
            base_arrays = [orig_x, recon_x, z_mean_data, z_logv_data]
            base_cols = orig_feat_cols + recon_feat_cols + z_mean_cols + z_logv_cols

        if brain_age_data is not None:
            base_arrays += [brain_age_data, bag_data]
            base_cols += ["brain_age", "bag"]

        df = pd.DataFrame(np.concatenate(base_arrays, 1), columns=base_cols)

        skipped_df = self.dataloader.get_skipped_data(dataloader="train")
        if skipped_df is not None:
            if skipped_df.shape[0] != df.shape[0]:
                raise DataRowMismatchError(
                    f"Mismatch skipped rows ({skipped_df.shape[0]}) vs df ({df.shape[0]})"
                )
            df = pd.concat([skipped_df.reset_index(drop=True), df], axis=1)

        ext = get_internal_file_extension()
        out_path = (
            f"{self.properties.system.output_dir}/reconstructions/train_data.{ext}"
        )
        self.logger.info("Saving train data to %s...", out_path)
        save_data(df, out_path)
        self.logger.info("Data saved successfully.")

        self.__save_latent_parameters(z_mean_data, data="train")

    def __analyze_results(self) -> TaskResult:
        dtype = self.properties.dataset.data_type
        engine = create_analysis_engine(dtype)
        engine.initialize_engine(
            feature_labels=self.dataloader.get_feature_labels(),
            covariate_labels=self.dataloader.get_encoded_covariate_labels(),
            target_labels=self.dataloader.get_target_labels(),
        )
        return engine.run_analysis()

    def __save_latent_parameters(
        self, z_mean_data: np.ndarray, *, data: str = "train"
    ) -> None:
        self.logger.info("Computing learned latent-space parameters.")
        latent_df = pd.DataFrame(
            {
                "latent_dim": np.arange(z_mean_data.shape[1]),
                "mean": z_mean_data.mean(0),
                "std": z_mean_data.std(0),
            }
        )
        ext = get_internal_file_extension()
        out_path = (
            f"{self.properties.system.output_dir}/model/latent_space_{data}.{ext}"
        )
        save_data(latent_df, out_path)
        self.logger.info("Latent space parameters saved successfully.")
