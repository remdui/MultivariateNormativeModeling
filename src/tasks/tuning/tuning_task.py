"""
Module for hyperparameter tuning using Optuna.

This module defines the TuningTask class, which runs hyperparameter tuning
for a model training task using the Optuna library.
"""

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
    """
    TuningTask performs hyperparameter tuning with Optuna by:

      - Defining an objective function that runs training with suggested hyperparameters.
      - Using a TPE sampler and Hyperband pruner for efficient search.
      - Reporting and storing the best hyperparameters and corresponding validation loss.
    """

    def __init__(self) -> None:
        """Initialize the TuningTask with a logger and set the task name."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TuningTask.")
        self.task_name = "tune"

    def run(self) -> TaskResult:
        """
        Run the hyperparameter tuning task.

        Configures Optuna's sampler and pruner, creates a study, and optimizes the
        objective function over a number of trials.

        Returns:
            TaskResult: Contains the best hyperparameters and the associated validation loss.
        """
        # Configure sampler and pruner for the study.
        sampler = TPESampler(seed=self.properties.general.seed, n_startup_trials=10)
        pruner = HyperbandPruner(min_resource=25, max_resource=200, reduction_factor=3)

        self.logger.info(
            f"Starting hyperparameter tuning using {sampler.__class__.__name__} + {pruner.__class__.__name__}."
        )

        # Create an Optuna study to minimize the validation loss.
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )
        study.optimize(self.objective, n_trials=200)

        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        # Optionally, retrain the final model with the best hyperparameters.
        # self._retrain_best_model(study.best_trial.params)

        return results

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna's optimization process.

        This function:
          - Clears and sets up a new experiment for each trial.
          - Suggests hyperparameters for the training task.
          - Updates properties accordingly.
          - Runs the training task and retrieves the validation loss.
          - Reports pruning if the trial is not promising.

        Args:
            trial (optuna.Trial): The current trial object.

        Returns:
            float: The validation loss for the current trial.
        """
        trial_id = trial.number

        # Reset experiment environment for a clean trial.
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_new_experiment(self.task_name)
        self.experiment_manager.add_experiment_group_identifier(str(trial_id))

        self.logger.info(f"Starting trial {trial_id}.")
        start_time = time.time()

        # Define fixed number of epochs for each trial.
        epochs = 500

        # Hyperparameter suggestions.
        latent_dim = 5
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        dropout = trial.suggest_categorical("dropout", [False, True])
        normalization = trial.suggest_categorical("normalization", [False, True])

        gradient_clipping = trial.suggest_categorical(
            "gradient_clipping", [False, True]
        )
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        activation_function = trial.suggest_categorical(
            "activation_function", ["silu", "relu", "leakyrelu"]
        )
        depth = trial.suggest_categorical("hidden_depth", [2, 3, 4])
        start_size = trial.suggest_categorical(
            "hidden_start_size", [512, 256, 128, 64, 32]
        )
        lr = trial.suggest_categorical("learning_rate", [0.001, 0.0001])
        beta_end = 1.0

        # Construct hidden layers based on depth and starting size.
        hidden_layers = [start_size]
        for _ in range(depth - 1):
            hidden_layers.append(hidden_layers[-1] // 2)

        self.logger.info(
            f"Trial {trial_id} hyperparameters:\n"
            f"  latent_dim: {latent_dim}\n"
            f"  batch_size: {batch_size}\n"
            f"  epochs: {epochs}\n"
            f"  dropout: {dropout}\n"
            f"  normalization: {normalization}\n"
            f"  gradient_clipping: {gradient_clipping}\n"
            f"  optimizer: {optimizer}\n"
            f"  activation_function: {activation_function}\n"
            f"  hidden_layers: {hidden_layers}\n"
            f"  learning_rate: {lr}"
            f"  beta_end: {beta_end}"
        )

        # Overwrite properties with suggested hyperparameters.
        properties = Properties.get_instance()
        properties.model.components["vae"]["latent_dim"] = latent_dim
        properties.train.batch_size = batch_size
        properties.train.epochs = epochs
        properties.train.gradient_clipping = gradient_clipping
        properties.train.optimizer = optimizer
        properties.model.dropout = dropout
        properties.model.normalization = normalization
        properties.model.hidden_layers = hidden_layers
        properties.train.loss_function_params["mse_vae"]["beta_end"] = beta_end
        properties.model.activation_function = activation_function
        if optimizer == "adam":
            properties.train.optimizer_params["adam"]["lr"] = lr
        else:
            properties.train.optimizer_params["adamw"]["lr"] = lr

        # Use an initializer suited for ReLU-based activations.
        if activation_function in {"relu", "leakyrelu"}:
            properties.model.weight_initializer = "he_normal"

        Properties.overwrite_instance(properties)

        # Run training with the current hyperparameters.
        train_task = TrainTask(
            trial=trial
        )  # TrainTask is assumed to support a trial callback.
        results = train_task.run()
        val_loss = results["reconstruction_loss"]["val_loss"]
        runtime = time.time() - start_time

        # Check for pruning: if the trial is not promising, Optuna can prune it.
        if trial.should_prune():
            self.logger.info(
                f"Trial {trial_id} pruned at val_loss={val_loss:.4f}, runtime={runtime:.2f} seconds."
            )
            raise optuna.TrialPruned()

        self.logger.info(
            f"Trial {trial_id} completed. Val_loss={val_loss:.4f}, runtime={runtime:.2f} seconds."
        )
        return val_loss

    def get_task_name(self) -> str:
        """
        Get the name of this task.

        Returns:
            str: The task name.
        """
        return self.task_name

    def _retrain_best_model(self, best_params: dict) -> None:
        """
        (Optional) Retrain the model using the best hyperparameters for reproducible final performance.

        Args:
            best_params (dict): The best hyperparameters found during tuning.
        """
        properties = Properties.get_instance()

        # Update properties with the best hyperparameters.
        properties.model.components["vae"]["latent_dim"] = best_params["latent_dim"]
        properties.train.loss_function_params["mse_vae"]["beta_end"] = best_params.get(
            "beta_end"
        )
        properties.train.optimizer_params["adam"]["lr"] = best_params["lr"]
        properties.train.batch_size = best_params["batch_size"]

        # Optionally, set a lower epoch count for the final run.
        properties.train.epochs = 50
        Properties.overwrite_instance(properties)

        final_train_task = TrainTask()
        final_results = final_train_task.run()

        self.logger.info("Final model retraining completed with best hyperparameters.")
        self.logger.info(
            f"Final model validation loss: {final_results['reconstruction_loss']['val_loss']:.4f}"
        )
