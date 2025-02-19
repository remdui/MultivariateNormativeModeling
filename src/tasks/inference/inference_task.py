"""
Module for model inference.

This module defines the InferenceTask class that executes the inference process,
including validating and processing results, saving metrics, and finalizing the experiment.
"""

from entities.log_manager import LogManager
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file


class InferenceTask(AbstractTask):
    """
    InferenceTask performs inference on a trained model.

    It validates and processes the results, writes metrics to file, and finalizes the experiment.
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

    def run(self) -> TaskResult:
        """
        Execute the inference process.

        This method:
          1. Validates the results.
          2. Processes the results.
          3. Writes the metrics to file.
          4. Finalizes the experiment.

        Returns:
            TaskResult: The result object containing inference metrics and any processed data.
        """
        results = TaskResult()
        results.validate_results()
        results.process_results()

        # Write processed metrics to the output file.
        write_results_to_file(results, "metrics")
        self.experiment_manager.finalize_experiment()

        return results
