"""Task for hyperparameter tuning using Optuna."""

import time

import optuna
from optuna.pruners import HyperbandPruner
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

        # https://optuna.readthedocs.io/en/stable/reference/samplers.html
        sampler = TPESampler(seed=self.properties.general.seed, n_startup_trials=10)

        # https://optuna.readthedocs.io/en/stable/reference/pruners.html
        pruner = HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3
        )

        # Create a study with a pruner and a sampler
        self.logger.info(
            f"Starting hyperparameter tuning using Optuna with {sampler.__class__.__name__} + {pruner.__class__.__name__})."
        )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )

        # Increase n_trials for a more comprehensive search
        study.optimize(self.objective, n_trials=100)

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

        # Set epochs for trial
        epochs = 50

        # Set hyperparameters to tune
        latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
        gradient_clipping = trial.suggest_categorical(
            "gradient_clipping", [False, True]
        )
        gradient_clipping_value = None
        if gradient_clipping:
            gradient_clipping_value = trial.suggest_float(
                "gradient_clipping_value", 1.0, 10.0
            )
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        activation_function = trial.suggest_categorical(
            "activation_function", ["silu", "relu", "leakyrelu"]
        )
        depth = trial.suggest_categorical("hidden_depth", [2, 3, 4])
        start_size = trial.suggest_categorical("hidden_start_size", [256, 128, 64])
        lr = trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001])

        # Construct hidden layers based on depth and start_size
        # For depth 2: [start_size, start_size/2]
        # For depth 3: [start_size, start_size/2, start_size/4]
        hidden_layers = [start_size]
        for _ in range(depth - 1):
            hidden_layers.append(hidden_layers[-1] // 2)

        # Log chosen hyperparameters
        self.logger.info(
            f"Trial {trial_id} hyperparameters:\n"
            f"  latent_dim={latent_dim}\n"
            f"  batch_size={batch_size}\n"
            f"  epochs={epochs}\n"
            f"  gradient_clipping={gradient_clipping}\n"
            f"  gradient_clipping_value={gradient_clipping_value}\n"
            f"  optimizer={optimizer}\n"
            f"  activation_function={activation_function}\n"
            f"  hidden_layers={hidden_layers}\n"
            f"  learning_rate={lr}"
        )

        # Overwrite properties
        properties = Properties.get_instance()

        properties.model.components["vae"]["latent_dim"] = latent_dim
        properties.train.batch_size = batch_size
        properties.train.epochs = epochs
        properties.train.gradient_clipping = gradient_clipping
        if gradient_clipping_value is not None:
            properties.train.gradient_clipping_value = gradient_clipping_value
        properties.train.optimizer = optimizer
        properties.model.hidden_layers = hidden_layers
        properties.model.activation_function = activation_function
        if optimizer == "adam":
            properties.train.optimizer_params["adam"]["lr"] = lr
        else:
            properties.train.optimizer_params["adamw"]["lr"] = lr

        # Set the weight initializer based on activation function
        if activation_function in {"relu", "leakyrelu"}:
            properties.model.weight_initializer = "he_normal"

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
