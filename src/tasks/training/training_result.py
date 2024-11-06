"""TrainingResult class."""

from entities.log_manager import LogManager
from tasks.abstract_result import AbstractResult


class TrainingResult(AbstractResult):
    """Class for training result data."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = LogManager.get_logger(__name__)

    def process_results(self) -> None:
        """Process the result data."""
        self.logger.info("Processing the training results.")

    def validate_results(self) -> None:
        """Validate the result data."""
        self.logger.info("Validating the training results.")
