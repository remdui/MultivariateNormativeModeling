"""Task for experiments."""

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
    """Task for experiment."""

    def __init__(self) -> None:
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ExperimentTask.")
        self.task_name = "experiment"

    def run(self) -> TaskResult:
        self.logger.info("Starting Experiment.")

        # Create a study with no pruning or random sampling
        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.properties.general.seed)
        )

        latent_dim_values = [1, 2, 3, 4, 5, 8, 16, 24, 32]

        # Force Optuna to perform one trial for each latent_dim value
        study.optimize(self.objective, n_trials=len(latent_dim_values))

        # Logging best trial results
        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        # Store results
        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        return results

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to iterate through all latent_dim values."""
        trial_id = trial.number
        self.logger.info(f"Starting trial {trial_id}.")
        start_time = time.time()
        latent_dim_values = [1, 2, 3, 4, 5, 8, 16, 24, 32]
        # Assign a specific latent_dim based on trial number
        latent_dim = latent_dim_values[trial_id]

        self.logger.info(
            f"Trial {trial_id} hyperparameters:\n  latent_dim={latent_dim}"
        )

        # Overwrite properties
        properties = Properties.get_instance()
        properties.model.components["vae"]["latent_dim"] = latent_dim
        Properties.overwrite_instance(properties)

        # Run the training task
        train_task = TrainTask()
        results = train_task.run()

        # Run the validation task
        val_task = ValidateTask()
        test_results = val_task.run()

        val_loss = results["reconstruction_loss"]["val_loss"]
        test_loss_r2 = test_results["r2"]
        test_loss_mse = test_results["mse"]
        runtime = time.time() - start_time

        self.logger.info(
            f"Trial {trial_id} completed. Val_loss={val_loss:.4f}, Test_R2{test_loss_r2:.4f}, Test_MSE{test_loss_mse:.4f}, Runtime={runtime:.2f} seconds."
        )
        return val_loss

    def get_task_name(self) -> str:
        """Get the task name.

        Returns:
            str: The task name.
        """
        return self.task_name
