"""Inference Task module."""

from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file


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
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)

    def run(self) -> TaskResult:
        """Run the inference task."""
        results = TaskResult()
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()
        return results
