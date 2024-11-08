"""Inference Task module."""

from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult


class InferenceTask(AbstractTask):
    """Inference Task class."""

    def __init__(self) -> None:
        """Initialize the Inference Task class."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing Inference Task.")

    def get_task_name(self) -> str:
        """Return the task name.

        Returns:
            str: The task name.
        """
        return "inference"

    def run(self) -> TaskResult:
        """Run the inference task."""
        results = TaskResult()
        return results
