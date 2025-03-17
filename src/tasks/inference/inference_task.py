"""
Module for model inference.

This module defines the InferenceTask class that executes the inference process,
including normalizing input features, running inference, and computing evaluation metrics.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from torch import autocast

from analysis.utils.normalization_reader import NormalizationReader
from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file
from util.model_utils import load_model


class InferenceTask(AbstractTask):
    """
    InferenceTask performs inference on a trained model.

    It normalizes the input, runs the model, and computes evaluation metrics.
    """

    def __init__(self) -> None:
        """Initialize the InferenceTask with logging and set up the inference experiment."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing Inference Task.")
        self.__init_inference_task()

    def __init_inference_task(self) -> None:
        """Set up the inference task by clearing any previous outputs and creating a new experiment."""
        self.task_name = "inference"
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)
        self.normalization_reader = NormalizationReader()
        self.model_file = self.properties.validation.model
        self.model_path = self.properties.system.models_dir + "/" + self.model_file
        self.model = load_model(self.model, self.model_path, self.device)
        self.model.eval()

    def run(self) -> TaskResult:
        """
        Execute the inference process.

        Returns:
            TaskResult: The result object containing inference metrics.
        """
        self.logger.info("Running inference.")

        real_values = np.array(
            [
                [
                    1200,
                    900,
                    1500,
                    800,
                    500,
                    300,
                    1200,
                    1400,
                    1000,
                    1300,
                    850,
                    1250,
                    1100,
                    950,
                    1050,
                    1120,
                    980,
                    560,
                    1150,
                    340,
                    730,
                    670,
                    920,
                    860,
                    1500,
                    1350,
                    770,
                    1400,
                    1600,
                    1350,
                    1100,
                    1240,
                    380,
                    1300,
                    1250,
                    890,
                    1480,
                    820,
                    480,
                    290,
                    1190,
                    1380,
                    980,
                    1280,
                    840,
                    1220,
                    1080,
                    940,
                    1040,
                    1110,
                    960,
                    550,
                    1140,
                    330,
                    720,
                    660,
                    910,
                    850,
                    1480,
                    1340,
                    760,
                    1390,
                    1590,
                    1340,
                    1090,
                    1230,
                    370,
                    500,
                ]
            ]
        )

        feature_names = [
            "lh_bankssts_vol",
            "lh_caudalanteriorcingulate_vol",
            "lh_caudalmiddlefrontal_vol",
            "lh_cuneus_vol",
            "lh_entorhinal_vol",
            "lh_frontalpole_vol",
            "lh_fusiform_vol",
            "lh_inferiorparietal_vol",
            "lh_inferiortemporal_vol",
            "lh_insula_vol",
            "lh_isthmuscingulate_vol",
            "lh_lateraloccipital_vol",
            "lh_lateralorbitofrontal_vol",
            "lh_lingual_vol",
            "lh_medialorbitofrontal_vol",
            "lh_middletemporal_vol",
            "lh_paracentral_vol",
            "lh_parahippocampal_vol",
            "lh_parsopercularis_vol",
            "lh_parsorbitalis_vol",
            "lh_parstriangularis_vol",
            "lh_pericalcarine_vol",
            "lh_postcentral_vol",
            "lh_posteriorcingulate_vol",
            "lh_precentral_vol",
            "lh_precuneus_vol",
            "lh_rostralanteriorcingulate_vol",
            "lh_rostralmiddlefrontal_vol",
            "lh_superiorfrontal_vol",
            "lh_superiorparietal_vol",
            "lh_superiortemporal_vol",
            "lh_supramarginal_vol",
            "lh_temporalpole_vol",
            "lh_transversetemporal_vol",
            "rh_bankssts_vol",
            "rh_caudalanteriorcingulate_vol",
            "rh_caudalmiddlefrontal_vol",
            "rh_cuneus_vol",
            "rh_entorhinal_vol",
            "rh_frontalpole_vol",
            "rh_fusiform_vol",
            "rh_inferiorparietal_vol",
            "rh_inferiortemporal_vol",
            "rh_insula_vol",
            "rh_isthmuscingulate_vol",
            "rh_lateraloccipital_vol",
            "rh_lateralorbitofrontal_vol",
            "rh_lingual_vol",
            "rh_medialorbitofrontal_vol",
            "rh_middletemporal_vol",
            "rh_paracentral_vol",
            "rh_parahippocampal_vol",
            "rh_parsopercularis_vol",
            "rh_parsorbitalis_vol",
            "rh_parstriangularis_vol",
            "rh_pericalcarine_vol",
            "rh_postcentral_vol",
            "rh_posteriorcingulate_vol",
            "rh_precentral_vol",
            "rh_precuneus_vol",
            "rh_rostralanteriorcingulate_vol",
            "rh_rostralmiddlefrontal_vol",
            "rh_superiorfrontal_vol",
            "rh_superiorparietal_vol",
            "rh_superiortemporal_vol",
            "rh_supramarginal_vol",
            "rh_temporalpole_vol",
            "rh_transversetemporal_vol",
        ]

        z_score_input = self.normalization_reader.transform_to_z_score(
            real_values, feature_names
        )

        input_tensor = torch.tensor(z_score_input, dtype=torch.float32).to(self.device)

        with torch.no_grad(), autocast(
            enabled=self.properties.train.mixed_precision, device_type=self.device
        ):
            model_output = self.model(
                input_tensor, None
            )  # Assuming model doesn't need covariates

        reconstructed_output = model_output["x_recon"].cpu().numpy()

        mse = mean_squared_error(real_values, reconstructed_output)
        r2 = r2_score(real_values, reconstructed_output)

        self.logger.info(f"Reconstruction MSE: {mse:.6f}")
        self.logger.info(f"Reconstruction R²: {r2:.6f}")

        results = TaskResult()
        results["mse"] = mse
        # TODO: Change this since r2 is not reliable on small sample (or n<=2) calculations
        results["r2"] = r2

        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()

        return results
