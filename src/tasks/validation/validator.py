"""Validator class module."""

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.validation.validation_result import ValidationResult


class Validator:
    """Validator class to validate the model."""

    def __init__(self) -> None:
        """Initialize the Validator class."""
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

    def run(self) -> ValidationResult:
        """Run the validation process."""
        self.logger.info("Starting the validation process.")
        return ValidationResult()
