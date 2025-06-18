"""Experiment task."""

import random
import time

import numpy as np
import optuna
import torch
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
        self.embedding_methods = [
            "no_embedding",
            "encoderdecoder_embedding",
            "fair_embedding",
            "harmonized",
        ]

        self.dataset_basename = "site_dataset_"
        self.dataset_ext = ".rds"
        self.haromized_prefix = "harmonized_"

        self.sites = [0, 1, 2]

        # Define latent dimension values to test
        self.latent_dim_values = [1, 2, 3, 4, 5, 8, 12, 16]
        # Number of repetitions per experiment setting
        self.num_repetitions = 3  # Change as needed

        self.experiment_manager.clear_output_directory()
        self.embed_method: str = ""
        self.trial_offset = 0
        self.rep_offset = 0
        # Base seed for the first repetition
        self.base_seed = 42
        # Keep the initial base seed to compute offsets per repetition
        self.initial_base_seed = self.base_seed
        self.seed = -1

    def set_seed_for_run(self, i: int, next_embed_method: str) -> None:
        """Set the seed for the i-th run, resetting per embedding method."""
        # Reset to base_seed when moving to a new embedding method
        if self.seed != self.base_seed and next_embed_method != self.embed_method:
            self.seed = self.base_seed

        # Increment seed for each run
        seed = self.seed + 1
        print(f"Run {i}: Using seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed = seed

    def run(self) -> TaskResult:
        """
        Execute the experiment by running structured trials across embedding methods,.

        latent dimensions, and repetitions.

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
                    for site in self.sites:
                        experiment_settings.append((embed, latent_dim, rep, site))

        # Run structured trials in the interleaved order
        study.optimize(
            lambda trial: self.objective(trial, experiment_settings),
            n_trials=len(experiment_settings),
        )

        # Log best trial information
        self.logger.info("Experiments completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        # Store best trial results
        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value

        return results

    def objective(self, trial: optuna.Trial, experiment_settings: list) -> float:
        """Task objective."""
        trial_id = trial.number + self.trial_offset
        embed_method, latent_dim, rep, site = experiment_settings[trial.number]

        # shift seed range per repetition
        self.base_seed = self.initial_base_seed + rep * len(self.latent_dim_values)
        self.set_seed_for_run(trial_id, embed_method)
        self.embed_method = embed_method

        # figure out which embedding to actually use in the model, and which dataset prefix
        if embed_method == "harmonized":
            actual_embed = "no_embedding"
            dataset_prefix = self.haromized_prefix + self.dataset_basename
        else:
            actual_embed = embed_method
            dataset_prefix = self.dataset_basename

        # e.g. "site_dataset_1_site_0.rds" or "harmonized_site_dataset_3_site_2.rds"
        filename = f"{dataset_prefix}{rep}_site_{site}{self.dataset_ext}"

        # set up the experiment folder name
        rep_display = rep + self.rep_offset
        embed_tag = embed_method.replace("_", "")
        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_experiment_group_identifier(
            f"{trial_id}_embed-{embed_tag}_dim-{latent_dim}"
            f"_rep-{rep_display}_seed-{self.seed}_testsite-{site}"
        )

        # update model config & dataset path in Properties
        props = Properties.get_instance()
        model_cfg = props.model.components.get(props.model.architecture, {})
        model_cfg["covariate_embedding"] = actual_embed
        model_cfg["latent_dim"] = latent_dim
        props.model.components[props.model.architecture] = model_cfg

        props.general.seed = self.seed
        props.dataset.input_data = filename
        Properties.overwrite_instance(props)

        self.__setup_task()

        start_time = time.time()
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
            f"Completed trial {trial_id} | Embedding = {embed_method} | Latent Dim = {latent_dim} | Repetition = {rep} | Site = {site}"
        )
        self.logger.info(
            f"| Val Loss = {val_loss:.4f} | Test R2 = {test_loss_r2:.4f} | Test MSE = {test_loss_mse:.4f}"
        )
        self.logger.info(f"| Runtime = {runtime:.2f} sec")
        self.experiment_manager.clear_experiment_group_identifier()
        return val_loss

    def get_task_name(self) -> str:
        """Retrieve the name of the task."""
        return self.task_name
