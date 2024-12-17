"""Task for hyperparameter tuning using Optuna."""

import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from entities.log_manager import LogManager
from entities.properties import Properties
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from tasks.training.train_task import TrainTask


class TuningTask(AbstractTask):
    """Task for hyperparameter tuning using Optuna."""

    def __init__(self) -> None:
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TrainTask.")
        self.task_name = "tune"

    def run(self) -> TaskResult:
        """Run the hyperparameter tuning task using Optuna.

        Returns:
            TaskResult: The task result object containing the best hyperparameters and validation loss.
        """
        # Advanced Optuna settings: Use TPE sampler and MedianPruner
        sampler = TPESampler(
            seed=42, n_startup_trials=10
        )  # https://optuna.readthedocs.io/en/stable/reference/samplers.html
        pruner = MedianPruner(
            n_startup_trials=5, n_warmup_steps=5
        )  # https://optuna.readthedocs.io/en/stable/reference/pruners.html

        # Create a study with a pruner and a sampler
        self.logger.info(
            "Starting hyperparameter tuning using Optuna (TPE Sampler + Median Pruner)."
        )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )

        # Increase n_trials for a more comprehensive search
        study.optimize(self.objective, n_trials=50, show_progress_bar=True)

        # Logging best trial results
        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        # Store results
        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        # Optional: Use best hyperparameters to retrain a final model
        # for stable and reproducible final performance. Example:
        # self.logger.info("Retraining best model with found hyperparameters...")
        # self._retrain_best_model(study.best_trial.params)

        return results

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize.

        Args:
            trial (optuna.Trial): The current trial object.

        Returns:
            float: The validation loss for the current trial.
        """
        trial_id = trial.number
        self.logger.info(f"Starting trial {trial_id}.")

        start_time = time.time()

        # Get the global properties instance
        properties = Properties.get_instance()

        # Define hyperparameter search space
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])
        beta_end = trial.suggest_uniform("beta_end", 0.1, 1.0)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        epochs = 20  # Can be tuned or fixed

        # Log chosen hyperparameters
        self.logger.info(
            f"Trial {trial_id} hyperparameters: "
            f"latent_dim={latent_dim}, beta_end={beta_end:.4f}, lr={lr:.6f}, "
            f"batch_size={batch_size}, epochs={epochs}"
        )

        # Overwrite properties
        properties.model.components["vae"]["latent_dim"] = latent_dim
        properties.train.loss_function_params["mse_vae"]["beta_end"] = beta_end
        properties.train.optimizer_params["adam"]["lr"] = lr
        properties.train.batch_size = batch_size
        properties.train.epochs = epochs

        properties.train.early_stopping.enabled = True
        properties.train.early_stopping.patience = 5

        Properties.overwrite_instance(properties)

        # Now run the training task with these updated properties
        # Assume TrainTask can accept a callback or we modify TrainTask to call `trial.report(val_loss, epoch)`
        # at the end of each epoch. If not, we only have final val_loss.

        # If TrainTask supports intermediate reporting:
        # Example modification in TrainTask: pass the trial object and do trial.report(val_loss, epoch).
        # Then Optuna can prune mid-training if val_loss isn't improving.

        train_task = TrainTask(trial=trial)  # If we modify TrainTask to accept trial
        results = train_task.run()

        val_loss = results["reconstruction_loss"]["val_loss"]

        runtime = time.time() - start_time

        # Check if the trial was pruned
        if trial.should_prune():
            self.logger.info(
                f"Trial {trial_id} is pruned at val_loss={val_loss:.4f}, Runtime={runtime:.2f} seconds."
            )
            raise optuna.TrialPruned()

        # Log final results for this trial
        self.logger.info(
            f"Trial {trial_id} completed. "
            f"Val_loss={val_loss:.4f}, Runtime={runtime:.2f} seconds."
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
