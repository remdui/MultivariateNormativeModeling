"""Task for hyperparameter tuning using Optuna."""

import time

import optuna
from optuna.samplers import TPESampler

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from tasks.training.train_task import TrainTask
from tasks.validation.validate_task import ValidateTask


class TuningTask(AbstractTask):
    """Task for hyperparameter tuning using Optuna."""

    def __init__(self) -> None:
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TrainTask.")
        self.task_name = "tune"

    def run(self) -> TaskResult:
        self.logger.info("Starting hyperparameter tuning using Optuna.")

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

    def _retrain_best_model(self, best_params: dict) -> None:
        """Optional: Retrain the model using the best hyperparameters for a stable final model."""
        properties = Properties.get_instance()
        # Overwrite properties with best_params
        properties.model.components["vae"]["latent_dim"] = best_params["latent_dim"]
        properties.train.loss_function_params["mse_vae"]["beta_end"] = best_params[
            "beta_end"
        ]
        properties.train.optimizer_params["adam"]["lr"] = best_params["lr"]
        properties.train.batch_size = best_params["batch_size"]

        # Potentially increase epochs for a final run
        properties.train.epochs = 50
        Properties.overwrite_instance(properties)

        final_train_task = TrainTask()
        final_results = final_train_task.run()
        self.logger.info(
            "Final model retraining completed with best found hyperparameters."
        )
        self.logger.info(
            f"Final model validation loss: {final_results['reconstruction_loss']['val_loss']:.4f}"
        )
