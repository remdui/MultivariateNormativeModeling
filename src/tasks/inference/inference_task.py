"""Module for model inference."""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from torch import Tensor, autocast

from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file
from util.model_utils import load_model


class InferenceTask(AbstractTask):
    """
    InferenceTask performs inference on a trained model.

    It runs the model on the configured test dataloader and computes reconstruction metrics.
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
        self.logger.info("Running inference on test data.")

        all_targets: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                data, covariates = batch
                data = data.to(self.device)
                covariates = covariates.to(self.device)

                with autocast(
                    enabled=self.mixed_precision_enabled, device_type=self.device
                ):
                    model_output = self.model(data, covariates)

                reconstructed_output = model_output["x_recon"]
                target = self.__resolve_inference_target(
                    reconstructed_output, data, covariates
                )

                all_targets.append(target.cpu().numpy())
                all_predictions.append(reconstructed_output.cpu().numpy())

        if not all_targets:
            raise ValueError(
                "Inference test dataloader is empty; no samples to evaluate."
            )

        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.concatenate(all_predictions, axis=0)

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.logger.info(f"Reconstruction MSE: {mse:.6f}")
        self.logger.info(f"Reconstruction R²: {r2:.6f}")

        results = TaskResult()
        results["mse"] = mse
        # TODO: Change this since r2 is not reliable on small sample (or n<=2) calculations
        results["r2"] = r2

        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()

        return results

    @staticmethod
    def __resolve_inference_target(
        reconstructed_output: Tensor, data: Tensor, covariates: Tensor
    ) -> Tensor:
        """Resolve the expected reconstruction target based on output dimensionality."""
        if (
            reconstructed_output.ndim == data.ndim
            and reconstructed_output.shape[:-1] == data.shape[:-1]
            and reconstructed_output.shape[-1] == data.shape[-1] + covariates.shape[-1]
        ):
            return torch.cat([data, covariates], dim=-1)
        return data
