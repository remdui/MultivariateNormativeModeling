"""Validator class module."""

from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.validation.validation_result import ValidationResult
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

        # Get the validation properties
        self.metrics = self.properties.validation.metrics
        self.data_representation = self.properties.validation.data_representation
        self.model_file = self.properties.validation.model
        self.baseline_model_file = self.properties.validation.baseline_model

        # Load model state dictionary from model file
        self.model_path = self.properties.system.models_dir + "/" + self.model_file
        self.model = load_model(self.model, self.model_path, self.device)

    def run(self) -> ValidationResult:
        """Run the validation process.

        Returns:
            ValidationResult: The validation result object.
        """
        self.logger.info("Starting the validation process.")

        return ValidationResult()
