"""Validator class module."""

import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor, autocast, no_grad
from tqdm import tqdm

from analysis.metrics.mse import compute_mse
from analysis.metrics.r2 import compute_r2_score
from analysis.visualization.image import combine_images, tensor_to_image
from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.data_utils import sample_batch_from_indices
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

        # Get the validation properties
        self.metrics = self.properties.validation.metrics
        self.data_representation = self.properties.validation.data_representation
        self.model_file = self.properties.validation.model
        self.model_path = self.properties.system.models_dir + "/" + self.model_file
        self.baseline_model_file = str(self.properties.validation.baseline_model)
        self.baseline_model_path = (
            self.properties.system.models_dir + "/" + self.baseline_model_file
        )

        # Load model state dictionary from model file
        self.model = load_model(self.model, self.model_path, self.device)

    def run(self) -> TaskResult:
        """Run the validation process using flattened R² scores.

        Returns:
            ValidationResult: The validation result object.
        """
        self.logger.info("Starting the validation process.")

        # Initialize results dictionary
        results = TaskResult()

        total_loss = 0.0
        total_samples = 0

        # Lists to accumulate predictions and targets
        all_preds = []
        all_targets = []

        # Process each batch in the test dataset
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Validating"):
                data, _ = batch
                data = data.to(self.device)

                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    recon_batch, z_mean, z_logvar = self.model(data)  # Forward pass
                    loss = self.loss(
                        recon_batch, data, z_mean, z_logvar
                    )  # Compute loss
                    total_loss += loss.item()
                    total_samples += data.size(0)

                    # Move tensors to CPU for metric calculation
                    preds_cpu = recon_batch.detach().cpu().numpy()
                    targets_cpu = data.detach().cpu().numpy()

                    # Accumulate predictions and targets
                    all_preds.append(preds_cpu)
                    all_targets.append(targets_cpu)

        # Concatenate all predictions and targets
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)

        # Convert to tensor
        all_preds_tensor = torch.from_numpy(all_preds_np)
        all_targets_tensor = torch.from_numpy(all_targets_np)

        # Compute MSE
        mse_value = compute_mse(
            all_targets_tensor, all_preds_tensor, metric_type="total"
        )

        # Compute R² score
        r2_value = compute_r2_score(
            all_targets_tensor, all_preds_tensor, metric_type="total"
        )

        avg_loss = total_loss / total_samples
        self.logger.info(f"Average test loss: {avg_loss:.4f}")
        self.logger.info(f"MSE: {mse_value:.4f}")
        self.logger.info(f"R²: {r2_value:.4f}")

        results["average_loss"] = avg_loss
        results["mse"] = mse_value.item()
        results["r2"] = r2_value.item()

        # Draw image samples if applicable
        if (
            self.data_representation == "image"
            or self.properties.dataset.data_type == "image"
        ):
            self.__draw_image_samples()
        elif (
            self.properties.dataset.data_type == "tabular"
            and self.data_representation == "tabular"
        ):
            self.__analyse_tabular_samples()

        return results

    def __draw_image_samples(self) -> None:
        """Draw original and reconstructed images if data is represented in a flattened tabular form."""
        self.logger.info(
            "Creating samples to compare original and reconstructed images."
        )

        # Select random indices from the test dataset
        num_samples = self.properties.validation.image.num_visual_samples
        total_samples = len(self.test_dataloader)
        random_indices = random.sample(range(total_samples), num_samples)

        # Dimensions for image reconstruction if data is flattened
        image_width = self.properties.validation.image.width
        image_length = self.properties.validation.image.length

        # Retrieve data for the selected indices
        original_images: list[Tensor] = []
        reconstructed_images: list[Tensor] = []

        # Access specific samples by index
        with torch.no_grad():
            data_batch = sample_batch_from_indices(
                self.test_dataloader.dataset, random_indices, self.device
            )

            # Apply autocast for mixed precision inference
            with autocast(
                enabled=self.properties.train.mixed_precision,
                device_type=self.device,
            ):
                recon_batch, _, _ = self.model(
                    data_batch
                )  # Forward pass for the entire batch

            # Append original and reconstructed images
            original_images.extend(
                data_batch.cpu().unbind()
            )  # Unbind to individual tensors
            reconstructed_images.extend(
                recon_batch.cpu().unbind()
            )  # Unbind to individual tensors

        for idx in range(num_samples):
            # Convert to PIL images
            original_image = tensor_to_image(
                original_images[idx], image_length, image_width
            )
            reconstructed_image = tensor_to_image(
                reconstructed_images[idx], image_length, image_width
            )

            # Combine original and reconstructed images side by side
            combined_image = combine_images(original_image, reconstructed_image)

            # Show images
            if self.properties.validation.image.show_image_samples:
                combined_image.show(title=f"Sample {idx + 1}")

            # Save images
            if self.properties.validation.image.save_image_samples:
                output_file_path = f"{self.properties.system.output_dir}/reconstructions/{self.model_name}_image_sample_{idx + 1}.png"
                combined_image.save(output_file_path)
                self.logger.info(f"Sample {idx + 1} saved to {output_file_path}")

    def __analyse_tabular_samples(self) -> None:
        """Analyze original and reconstructed tabular data samples and compare them.

        This method:
        - Randomly selects a set of rows from the test dataset.
        - Runs these rows through the model to get their reconstructed values.
        - Prints out original vs reconstructed values side by side.
        - Applies and prints some metrics (e.g., MSE) between the original and reconstruction.
        - Ensures that all columns are visible in the printout by adjusting pandas options.

        Notes:
            - Assumes the test dataset returns tabular data in a tensor format.
            - Assumes `sample_batch_from_indices` and the model are compatible with tabular data.
            - Assumes that the dataset or properties object contains information on how to interpret the data.
        """

        self.logger.info(
            "Analyzing samples to compare original and reconstructed tabular data."
        )

        num_samples = 1
        total_samples = len(self.test_dataloader)
        random_indices = random.sample(range(total_samples), num_samples)

        # Retrieve data for the selected indices
        with no_grad():
            data_batch = sample_batch_from_indices(
                self.test_dataloader.dataset, random_indices, self.device
            )

            with autocast(
                enabled=self.properties.train.mixed_precision,
                device_type=self.device,
            ):
                reconstructed_batch, _, _ = self.model(data_batch)

        # Convert tensors to CPU for processing
        original_np = data_batch.cpu().numpy()
        reconstructed_np = reconstructed_batch.cpu().numpy()

        # Set pandas display options to ensure all columns are visible
        orig_max_cols = pd.get_option("display.max_columns")
        pd.set_option("display.max_columns", None)

        try:
            # Convert to DataFrames for nice printing
            original_df = pd.DataFrame(
                original_np, columns=self.dataloader.get_feature_names()
            )
            reconstructed_df = pd.DataFrame(
                reconstructed_np, columns=self.dataloader.get_feature_names()
            )

            # Compute metrics (e.g., MSE for each sample and overall)
            mse_values = ((original_df - reconstructed_df) ** 2).mean(axis=1)
            overall_mse = mse_values.mean()

            self.logger.info("Original vs. Reconstructed Tabular Samples:")
            for i in range(num_samples):
                self.logger.info(f"Sample Index: {random_indices[i]}")
                self.logger.info("Original:")
                self.logger.info(f"\n{str(original_df.iloc[i : i + 1])}")
                self.logger.info("Reconstructed:")
                self.logger.info(f"\n{str(reconstructed_df.iloc[i : i + 1])}")

                sample_mse = mse_values.iloc[i]
                self.logger.info(f"MSE for this sample: {sample_mse:.6f}")

            self.logger.info(f"Average MSE across selected samples: {overall_mse:.6f}")

        finally:
            # Reset pandas display options back to original
            pd.set_option("display.max_columns", orig_max_cols)
