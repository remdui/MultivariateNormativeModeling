"""
Module for running structured experiments.

This module defines the ExperimentTask class which:
  - Iterates over different embedding techniques.
  - Iterates over a predefined set of latent dimension values.
  - Repeats the experiment for a given number of repetitions.
  - Runs a training and validation cycle for each configuration.
  - Uses Optuna to track trials, ensuring structured hyperparameter search.
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
    ExperimentTask runs a series of experiments across different embedding methods.

    and latent dimensions with repetitions.

    The task interleaves execution across embedding methods for faster diversity of results.
    """

    def __init__(self) -> None:
        """Initialize the ExperimentTask with a logger and set the task name."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ExperimentTask.")
        self.task_name = "experiment"

        # Define embedding techniques to test
        self.embedding_methods = ["no_embedding", "input_feature", "encoder_embedding"]

        # Define latent dimension values to test
        self.latent_dim_values = [1, 2, 3, 4, 5, 8, 12, 16, 20]

        # Number of repetitions per experiment setting
        self.num_repetitions = 3  # Change as needed

        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)

    def run(self) -> TaskResult:
        """
        Execute the experiment by running structured trials across embedding methods, latent dimensions, and repetitions.

        Uses Optuna to track and manage the search space.

        Returns:
            TaskResult: A result object containing the best hyperparameters and corresponding validation loss.
        """
        self.logger.info("Starting structured Experiment.")

        # Create Optuna study for tracking trials
        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.properties.general.seed)
        )

        # Create experiment settings by interleaving embedding methods, latent dims, and repetitions
        experiment_settings = []
        for rep in range(self.num_repetitions):
            for embed in self.embedding_methods:
                for latent_dim in self.latent_dim_values:
                    experiment_settings.append((embed, latent_dim, rep))

        # Run structured trials in the interleaved order
        study.optimize(
            lambda trial: self.objective(trial, experiment_settings),
            n_trials=len(experiment_settings),
        )

        # Log best trial information
        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        # Store best trial results
        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        return results

    def objective(self, trial: optuna.Trial, experiment_settings: list) -> float:
        """
        Objective function for a structured experiment trial.

        Runs training and validation for a specific embedding method and latent dimension.

        Args:
            trial (optuna.Trial): The current Optuna trial object.
            experiment_settings (list): The ordered list of experiments to run.

        Returns:
            float: The validation loss from the training task.
        """
        # Assign experiment parameters from structured list
        trial_id = trial.number
        embed_method, latent_dim, repetition = experiment_settings[trial_id]
        # Clear previous outputs and set unique identifier for tracking
        self.experiment_manager.clear_output_directory()
        # self.experiment_manager.add_experiment_group_identifier(f"{trial_id}_embed-{embed_method}_dim-{latent_dim}_rep-{repetition}")

        self.logger.info(
            f"Starting trial {trial_id} | Embedding = {embed_method} | Latent Dim = {latent_dim} | Repetition = {repetition}"
        )

        start_time = time.time()

        # Update properties with the selected embedding method and latent dimension
        properties = Properties.get_instance()
        model_config = properties.model.components.get(
            properties.model.architecture, {}
        )
        model_config["covariate_embedding"] = embed_method
        model_config["latent_dim"] = latent_dim
        properties.model.components[properties.model.architecture] = model_config

        # Overwrite instance with new properties
        Properties.overwrite_instance(properties)

        # Run training task
        train_task = TrainTask()
        train_results = train_task.run()

        # Run validation task
        val_task = ValidateTask()
        test_results = val_task.run()

        # Extract key metrics
        val_loss = train_results["reconstruction_loss"]["val_loss"]
        test_loss_r2 = test_results["recon_r2"]
        test_loss_mse = test_results["recon_mse"]
        runtime = time.time() - start_time

        self.logger.info(
            f"Completed trial {trial_id} | Embedding = {embed_method} | Latent Dim = {latent_dim} | Repetition = {repetition}"
        )
        self.logger.info(
            f"| Val Loss = {val_loss:.4f} | Test R2 = {test_loss_r2:.4f} | Test MSE = {test_loss_mse:.4f}"
        )
        self.logger.info(f"| Runtime = {runtime:.2f} sec")

        return val_loss

    def get_task_name(self) -> str:
        """Retrieve the name of the task."""
        return self.task_name
