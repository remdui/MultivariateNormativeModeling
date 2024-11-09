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
        self.__init_inference_task()

    def __init_inference_task(self) -> None:
        """Setup the inference task."""
        self.task_name = "inference"

    def run(self) -> TaskResult:
        """Run the inference task."""
        results = TaskResult()
        return results
