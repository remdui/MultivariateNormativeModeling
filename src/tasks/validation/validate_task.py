"""Validator class module."""

import random

import torch
from tqdm import tqdm

from analysis.visualization.image import combine_images, tensor_to_image
from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
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
        """Run the validation process.

        Returns:
            ValidationResult: The validation result object.
        """
        self.logger.info("Starting the validation process.")

        # Initialize results dictionary
        results = TaskResult()

        total_loss = 0.0
        total_samples = 0

        # Process each batch in the test dataset
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Validating"):
                data, _ = batch
                data = data.to(self.device)

                # Perform a forward pass
                recon_batch, z_mean, z_logvar = self.model(data)

                # Calculate loss for the batch
                loss = self.loss(recon_batch, data, z_mean, z_logvar)
                total_loss += loss.item()
                total_samples += self.properties.train.batch_size

            # Calculate average loss
            avg_loss = total_loss / total_samples
            self.logger.info(f"Average validation loss: {avg_loss:.4f}")
            results["average_loss"] = avg_loss

        # Draw image samples if applicable
        if (
            self.data_representation == "image"
            or self.properties.dataset.data_type == "image"
        ):
            self.__draw_image_samples()

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
        original_images = []
        reconstructed_images = []

        # Access specific samples by index
        with torch.no_grad():
            for idx in tqdm(random_indices, desc="Image Sampling"):
                data, _ = self.test_dataloader.dataset[idx]  # Direct access to dataset
                data = data.to(self.device)  # Add batch dimension

                # Forward pass to get the reconstructed image
                recon_data, _, _ = self.model(data)

                # Append original and reconstructed images
                original_images.append(data.cpu())  # Remove batch dimension
                reconstructed_images.append(recon_data.cpu())  # Remove batch dimension

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
                output_file_path = f"{self.properties.system.output_dir}/{self.model_name}_image_sample_{idx + 1}.png"
                combined_image.save(output_file_path)
                self.logger.info(f"Sample {idx + 1} saved to {output_file_path}")
