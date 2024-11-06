"""This module contains the ValidationResult class."""

from entities.log_manager import LogManager
from tasks.common.abstract_result import AbstractResult


class ValidationResult(AbstractResult):
    """Class for validation result data."""

    def __init__(self) -> None:
        """Initialize the ValidationResult."""
        super().__init__()
        self.logger = LogManager.get_logger(__name__)

    def process_results(self) -> None:
        """Process the result data."""
        self.logger.info("Processing the validation results.")

    def validate_results(self) -> None:
        """Validate the result data."""
        self.logger.info("Validating the validation results.")
