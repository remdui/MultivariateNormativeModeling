"""
Module for running experiments.

This module defines the ExperimentTask class which:
  - Iterates over a predefined set of latent dimension values.
  - Runs a training and validation cycle for each latent dimension.
  - Uses Optuna's study mechanism (without pruning) to manage trials.
  - Logs and returns the best hyperparameters based on the validation loss.
"""

import time

import optuna
from optuna.samplers import TPESampler

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from tasks.training.train_task import TrainTask
from tasks.validation.validate_task import ValidateTask


class ExperimentTask(AbstractTask):
    """
    ExperimentTask runs a series of experiments with different latent dimensions.

    It forces a single trial for each latent dimension value, updating the model's properties,
    and then executing training and validation tasks to compute the performance metrics.
    """

    def __init__(self) -> None:
        """Initialize the ExperimentTask with a logger and set the task name."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ExperimentTask.")
        self.task_name = "experiment"

    def run(self) -> TaskResult:
        """
        Execute the experiment by running trials with different latent dimension values.

        Uses Optuna's study (with a TPESampler) to perform a trial for each latent_dim value.

        Returns:
            TaskResult: A result object containing the best hyperparameters and the corresponding validation loss.
        """
        self.logger.info("Starting Experiment.")

        # Initialize an Optuna study with TPESampler for reproducibility.
        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.properties.general.seed)
        )

        # Define a list of latent_dim values to test.
        latent_dim_values = [1, 2, 3, 4, 5, 8, 16, 24, 32]

        # Force one trial per latent_dim value.
        study.optimize(self.objective, n_trials=len(latent_dim_values))

        # Log best trial information.
        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        # Store best trial results.
        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        return results

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for a trial in the experiment.

        For each trial:
          1. Clears the experiment output.
          2. Selects a latent_dim value based on the trial number.
          3. Overwrites the model properties with the new latent_dim.
          4. Runs training and validation tasks.
          5. Logs and returns the validation loss.

        Args:
            trial (optuna.Trial): The current trial object.

        Returns:
            float: The validation loss from the training task.
        """
        trial_id = trial.number

        # Clear previous outputs and assign a unique group identifier for the trial.
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.add_experiment_group_identifier(str(trial_id))

        self.logger.info(f"Starting trial {trial_id}.")
        start_time = time.time()

        # Define the set of latent dimensions to iterate over.
        latent_dim_values = [1, 2, 3, 4, 5, 8, 16, 24, 32]
        # Assign latent_dim based on the trial number.
        latent_dim = latent_dim_values[trial_id]

        self.logger.info(
            f"Trial {trial_id} hyperparameters:\n  latent_dim = {latent_dim}"
        )

        # Update the properties with the selected latent dimension.
        properties = Properties.get_instance()
        properties.model.components["vae"]["latent_dim"] = latent_dim
        Properties.overwrite_instance(properties)

        # Run the training task with updated properties.
        train_task = TrainTask()
        train_results = train_task.run()

        # Run the validation task after training.
        val_task = ValidateTask()
        test_results = val_task.run()

        # Retrieve validation loss from training results.
        val_loss = train_results["reconstruction_loss"]["val_loss"]
        # Optionally, additional metrics can be logged from test_results.
        test_loss_r2 = test_results["r2"]
        test_loss_mse = test_results["mse"]
        runtime = time.time() - start_time

        self.logger.info(
            f"Trial {trial_id} completed. "
            f"Val_loss = {val_loss:.4f}, Test_R2 = {test_loss_r2:.4f}, Test_MSE = {test_loss_mse:.4f}, "
            f"Runtime = {runtime:.2f} seconds."
        )

        return val_loss

    def get_task_name(self) -> str:
        """
        Retrieve the name of the task.

        Returns:
            str: The task name.
        """
        return self.task_name
