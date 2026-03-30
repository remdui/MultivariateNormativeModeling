"""Experiment task."""

import random
import time
from pathlib import Path
from typing import Any

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
    """Run structured experiments over embedding methods, latent dimensions, and datasets."""

    def __init__(self) -> None:
        """Initialize the ExperimentTask with generic, config-driven settings."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing ExperimentTask.")
        self.task_name = "experiment"

        architecture = self.properties.model.architecture
        model_cfg = self.properties.model.components.get(architecture, {})
        matrix_cfg = self.properties.model.experiment_matrix
        default_embedding = model_cfg.get("covariate_embedding", "no_embedding")
        default_latent_dim = model_cfg.get("latent_dim", 32)
        embedding_source = matrix_cfg.embedding_methods
        latent_source = matrix_cfg.latent_dims
        dataset_source = matrix_cfg.dataset_files
        repetitions_source = matrix_cfg.repetitions

        self.embedding_methods = self.__as_str_list(
            embedding_source, fallback=[default_embedding]
        )
        self.latent_dim_values = self.__as_int_list(
            latent_source, fallback=[default_latent_dim]
        )
        self.dataset_variants = self.__as_str_list(
            dataset_source,
            fallback=[self.properties.dataset.input_data],
        )
        self.num_repetitions = max(1, int(repetitions_source))

        self.initial_base_seed = self.properties.general.seed
        self.seed = self.initial_base_seed
        self.experiment_manager.clear_output_directory()

        self.logger.info(
            "Experiment settings | embeddings=%s | latent_dims=%s | datasets=%s | repetitions=%d",
            self.embedding_methods,
            self.latent_dim_values,
            self.dataset_variants,
            self.num_repetitions,
        )

    @staticmethod
    def __as_str_list(value: Any, fallback: list[str]) -> list[str]:
        """Coerce a config value into a non-empty list of strings."""
        if isinstance(value, str):
            parsed = [value]
        elif isinstance(value, (list, tuple, set)):
            parsed = [str(item) for item in value if str(item).strip()]
        else:
            parsed = []
        unique_values = list(dict.fromkeys(parsed))
        return unique_values or fallback

    @staticmethod
    def __as_int_list(value: Any, fallback: list[int | str]) -> list[int]:
        """Coerce a config value into a non-empty list of integers."""
        parsed: list[int] = []
        source_values = value if isinstance(value, (list, tuple, set)) else [value]

        for item in source_values:
            try:
                if item is not None:
                    parsed.append(int(item))
            except (TypeError, ValueError):
                continue

        if not parsed:
            parsed = [int(item) for item in fallback]

        return list(dict.fromkeys(parsed))

    def set_seed_for_run(self, seed: int) -> None:
        """Set random seed for Python, NumPy, and PyTorch for one experiment run."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.properties.system.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.seed = seed

    def run(self) -> TaskResult:
        """
        Execute structured experiments.

        Returns:
            TaskResult: Best trial parameters and corresponding validation loss.
        """
        self.logger.info("Starting structured experiment.")

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.properties.general.seed)
        )

        experiment_settings: list[tuple[str, int, int, str]] = []
        for rep in range(self.num_repetitions):
            for embed in self.embedding_methods:
                for latent_dim in self.latent_dim_values:
                    for dataset_file in self.dataset_variants:
                        experiment_settings.append(
                            (embed, latent_dim, rep, dataset_file)
                        )

        study.optimize(
            lambda trial: self.objective(trial, experiment_settings),
            n_trials=len(experiment_settings),
        )

        self.logger.info("Experiments completed.")
        self.logger.info(f"Best trial parameters: {study.best_trial.params}")
        self.logger.info(f"Best trial value (val_loss): {study.best_trial.value:.4f}")

        results = TaskResult()
        results["best_params"] = study.best_trial.params
        results["best_val_loss"] = study.best_trial.value
        return results

    def objective(
        self, trial: optuna.Trial, experiment_settings: list[tuple[str, int, int, str]]
    ) -> float:
        """Run one configured experiment setting and return validation loss."""
        trial_id = trial.number
        embed_method, latent_dim, rep, dataset_file = experiment_settings[trial.number]

        seed = self.initial_base_seed + trial_id
        self.set_seed_for_run(seed)

        dataset_tag = Path(dataset_file).stem.replace("_", "-")
        embed_tag = embed_method.replace("_", "")

        self.experiment_manager.clear_output_directory()
        self.experiment_manager.create_experiment_group_identifier(
            f"{trial_id}_embed-{embed_tag}_dim-{latent_dim}"
            f"_rep-{rep}_seed-{seed}_dataset-{dataset_tag}"
        )

        props = Properties.get_instance()
        model_cfg = props.model.components.get(props.model.architecture, {})
        model_cfg["covariate_embedding"] = embed_method
        model_cfg["latent_dim"] = latent_dim
        props.model.components[props.model.architecture] = model_cfg

        props.general.seed = seed
        props.dataset.input_data = dataset_file
        Properties.overwrite_instance(props)

        self.rebuild_task_components()

        start_time = time.time()
        train_task = TrainTask()
        train_results = train_task.run()

        val_task = ValidateTask()
        test_results = val_task.run()

        val_loss = self.__extract_val_loss(train_results)
        test_loss_r2 = test_results["recon_r2"]
        test_loss_mse = test_results["recon_mse"]
        runtime = time.time() - start_time

        self.logger.info(
            "Completed trial %d | embedding=%s | latent_dim=%d | repetition=%d | dataset=%s",
            trial_id,
            embed_method,
            latent_dim,
            rep,
            dataset_file,
        )
        self.logger.info(
            "Metrics | val_loss=%.4f | test_r2=%.4f | test_mse=%.4f | runtime=%.2f sec",
            val_loss,
            test_loss_r2,
            test_loss_mse,
            runtime,
        )
        self.experiment_manager.clear_experiment_group_identifier()
        return val_loss

    @staticmethod
    def __extract_val_loss(train_results: TaskResult) -> float:
        """Extract validation loss from standard or cross-validation training results."""
        result_data = train_results.get_data()
        reconstruction_loss = result_data.get("reconstruction_loss")
        if reconstruction_loss and "val_loss" in reconstruction_loss:
            return float(reconstruction_loss["val_loss"])

        fold_losses = [
            float(value["val_loss"])
            for key, value in result_data.items()
            if key.startswith("reconstruction_loss_fold_")
            and isinstance(value, dict)
            and "val_loss" in value
        ]
        if fold_losses:
            return float(np.mean(fold_losses))

        raise ValueError("Validation loss not found in training results.")

    def get_task_name(self) -> str:
        """Retrieve the name of the task."""
        return self.task_name
