"""
Train the model using the configuration.

This module defines the TrainTask class which handles model training,
including optimizer and scheduler setup, weight initialization, early stopping,
cross-validation, mixed precision training, gradient accumulation, checkpointing,
and reporting of training metrics.
"""

from typing import Any

import optuna
import torch
from torch import GradScaler, Tensor, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from entities.log_manager import LogManager
from model.layers.initialization.factory import initialize_weights
from optimization.optimizers.factory import get_optimizer
from optimization.regularizers.early_stopping import EarlyStopping
from optimization.schedulers.factory import get_scheduler
from preprocessing.transform.impl.encoding import EncodingTransform
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file
from util.model_utils import save_model, visualize_model_arch


class TrainTask(AbstractTask):
    """
    TrainTask handles the training of the model based on configuration settings.

    This includes:
      - Setting up optimizer, scheduler, and automatic mixed precision (AMP).
      - Initializing model weights and early stopping.
      - Running training and validation loops, with support for cross-validation,
        gradient accumulation, and checkpointing.
    """

    def __init__(self, trial: optuna.Trial | None = None) -> None:
        """
        Initialize the TrainTask.

        Args:
            trial (optuna.Trial | None): Optional trial for hyperparameter tuning.
        """
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TrainTask.")
        self.__init_train_task()
        self.trial = trial
        self.experiment_manager.clear_output_directory()

    def __init_train_task(self) -> None:
        """Set up the training task components and report enabled optimizations."""
        self.task_name = "train"
        self.experiment_manager.create_new_experiment(self.task_name)
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__setup_amp()
        self.__initialize_weights()
        self.__initialize_early_stopping()
        self.__report_enabled_optimizations()

    def __reinitialize_train_task(self) -> None:
        """Reinitialize components for subsequent cross-validation folds."""
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__initialize_weights()
        self.early_stopping.reset()

    def __setup_optimizer(self) -> None:
        """
        Initialize the optimizer based on configuration.

        Retrieves the optimizer name and parameters, then creates an optimizer.
        """
        optimizer_name = self.properties.train.optimizer
        optimizer_params = self.properties.train.optimizer_params.get(
            optimizer_name, {}
        )
        self.optimizer = get_optimizer(
            optimizer_name, self.model.parameters(), **optimizer_params
        )
        self.logger.info(
            f"Initialized optimizer {optimizer_name} with parameters: {self.optimizer.state_dict()}"
        )

    def __setup_scheduler(self) -> None:
        """
        Initialize the learning rate scheduler based on configuration.

        Sets up the scheduler and determines if stepping occurs per batch.
        """
        scheduler_method = self.properties.train.scheduler
        scheduler_params = self.properties.train.scheduler_params.get(
            scheduler_method, {}
        )
        self.scheduler = get_scheduler(
            scheduler_method, self.optimizer, **scheduler_params
        )
        # Schedulers like cyclic, onecycle, or cosineannealingwarmrestarts step per batch.
        self.scheduler_step_per_batch = self.properties.train.scheduler in {
            "cyclic",
            "onecycle",
            "cosineannealingwarmrestarts",
        }
        self.logger.info(
            f"Initialized scheduler {scheduler_method} with parameters: {self.scheduler.state_dict()}"
        )
        self.logger.info(f"Scheduler steps per batch: {self.scheduler_step_per_batch}")

    def __setup_amp(self) -> None:
        """Initialize automatic mixed precision (AMP) for training."""
        self.scaler = GradScaler()

    def __initialize_weights(self) -> None:
        """
        Initialize model weights.

        Resets model parameters and applies the weight initialization method specified
        in the configuration.
        """
        # Reset parameters for layers that implement reset_parameters.
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        weight_initializer = self.properties.model.weight_initializer
        initialize_weights(self.model, weight_initializer)
        self.logger.info(f"Initialized weights using {weight_initializer}")

    def __initialize_early_stopping(self) -> None:
        """Initialize the early stopping mechanism."""
        self.early_stopping = EarlyStopping()

    def run(self) -> TaskResult:
        """
        Execute the training process.

        Handles standard training or cross-validation, saves the final model,
        writes results to file, visualizes the model architecture, and finalizes the experiment.

        Returns:
            TaskResult: Object containing training metrics and loss values.
        """
        results = TaskResult()
        epochs = self.properties.train.epochs

        self.logger.info(
            f"Training model: {self.model_name} for {epochs} epochs with batch size {self.batch_size} using device {self.device}"
        )

        if self.properties.train.cross_validation:
            self.__run_cross_validation_training(epochs, results)
        else:
            self.__run_training(epochs, results)

        # Save the final model if enabled.
        if self.properties.train.save_model:
            save_model(
                model=self.model,
                save_dir=self.model_save_dir,
                model_name=self.model_name,
                use_date=False,
            )

        self.logger.info("Training completed.")
        results.validate_results()
        results.process_results()
        write_results_to_file(results, "metrics")
        visualize_model_arch(self.get_model(), self.get_input_size())
        self.experiment_manager.finalize_experiment()

        # Save normalization statistics
        EncodingTransform.save_stats_to_file()

        return results

    def __run_cross_validation_training(
        self, epochs: int, results: TaskResult
    ) -> TaskResult:
        """
        Run cross-validation training across multiple folds.

        Args:
            epochs (int): Number of epochs per fold.
            results (TaskResult): Object to store training results.

        Returns:
            TaskResult: Updated results with cross-validation metrics.
        """
        for fold in range(self.properties.train.cross_validation_folds):
            self.logger.info(
                f"Starting fold {fold + 1} of {self.properties.train.cross_validation_folds}"
            )
            # Retrieve fold-specific dataloaders.
            train_dataloader, val_dataloader = self.dataloader.fold_dataloader(fold)
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

            # Reinitialize for folds beyond the first.
            if fold > 0:
                self.__reinitialize_train_task()

            self.__run_training(epochs, results, fold + 1)

        return results

    def __run_training(
        self, epochs: int, results: TaskResult, fold: int | None = None
    ) -> TaskResult:
        """
        Run the main training loop.

        Args:
            epochs (int): Number of epochs to train.
            results (TaskResult): Object to store training metrics.
            fold (int | None): Fold number if using cross-validation, None otherwise.

        Returns:
            TaskResult: Updated results after training.
        """
        current_learning_rate = self.scheduler.get_last_lr()[0]
        best_val_loss = float("inf")  # Initialize best validation loss.

        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()  # Set model to training mode.
            avg_loss = self.__process_training_epoch(epoch, epochs)
            avg_val_loss = self.__process_validation_epoch(epoch)

            # Report intermediate results to Optuna if applicable.
            # if self.trial is not None and epoch > 50:  # Skip warm-up period.
            #     self.trial.report(avg_val_loss, step=epoch)
            #     if self.trial.should_prune():
            #         self.logger.info(
            #             f"Trial {self.trial.number} pruned at epoch {epoch + 1} with val_loss={avg_val_loss:.4f}"
            #         )
            #         raise optuna.TrialPruned()

            best_val_loss = self.__update_best_model(avg_val_loss, best_val_loss)
            self.__store_results(results, fold, avg_loss, avg_val_loss)
            current_learning_rate = self.__step_scheduler(
                current_learning_rate, avg_val_loss
            )

            if self.__check_early_stopping(avg_val_loss):
                break

            self.__save_checkpoint(epoch)

        return results

    def __process_training_epoch(self, epoch: int, epochs: int) -> float:
        """
        Process a single training epoch.

        Args:
            epoch (int): Current epoch number.
            epochs (int): Total number of epochs.

        Returns:
            float: Average training loss for the epoch.
        """
        tqdm_loader = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        avg_loss = self.__train_epoch(tqdm_loader)
        self.logger.info(
            f"Average training loss after epoch {epoch + 1}: {avg_loss:.4f}"
        )
        return avg_loss

    def __process_validation_epoch(self, epoch: int) -> float:
        """
        Process validation over the current epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average validation loss.
        """
        avg_val_loss = self.__validate()
        self.logger.info(
            f"Average validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}"
        )
        return avg_val_loss

    def __update_best_model(self, avg_val_loss: float, best_val_loss: float) -> float:
        """
        Update and save the best model if the current validation loss improves.

        Args:
            avg_val_loss (float): Current epoch's average validation loss.
            best_val_loss (float): Best recorded validation loss so far.

        Returns:
            float: Updated best validation loss.
        """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(
                model=self.model,
                save_dir=self.model_save_dir,
                model_name=f"{self.model_name}_best",
                use_date=False,
            )
            self.logger.info(
                f"New best model saved with validation loss: {best_val_loss:.4f}"
            )
        return best_val_loss

    def __store_results(
        self,
        results: TaskResult,
        fold: int | None,
        avg_loss: float,
        avg_val_loss: float,
    ) -> None:
        """
        Store the training and validation results in the TaskResult.

        Args:
            results (TaskResult): Object to store results.
            fold (int | None): Fold number if cross-validation is used.
            avg_loss (float): Average training loss.
            avg_val_loss (float): Average validation loss.
        """
        if fold is not None:
            results[f"reconstruction_loss_fold_{fold}"] = {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
            }
        else:
            results["reconstruction_loss"] = {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
            }

    def __step_scheduler(
        self, current_learning_rate: float, avg_val_loss: float
    ) -> float:
        """
        Step the scheduler if not configured for per-batch stepping.

        Args:
            current_learning_rate (float): Current learning rate.
            avg_val_loss (float): Average validation loss.

        Returns:
            float: Updated learning rate.
        """
        if not self.scheduler_step_per_batch:
            if self.properties.train.scheduler == "plateau":
                self.scheduler.step(avg_val_loss)  # type: ignore
            else:
                self.scheduler.step()
            new_lr = self.scheduler.get_last_lr()[0]
            if current_learning_rate != new_lr:
                current_learning_rate = new_lr
                self.logger.info(f"Scheduler adjusted the learning rate to: {new_lr}")
        return current_learning_rate

    def __check_early_stopping(self, avg_val_loss: float) -> bool:
        """
        Check whether early stopping conditions are met.

        Args:
            avg_val_loss (float): Average validation loss.

        Returns:
            bool: True if early stopping is triggered, else False.
        """
        if (
            self.properties.train.early_stopping.enabled
            and self.early_stopping.stop_condition_met(avg_val_loss)
        ):
            self.logger.info("Early stopping triggered.")
            return True
        return False

    def __save_checkpoint(self, epoch: int) -> None:
        """
        Save a model checkpoint if enabled and at the defined interval.

        Args:
            epoch (int): Current epoch number.
        """
        if (
            self.properties.train.checkpoint.save_checkpoint
            and (epoch + 1) % self.properties.train.checkpoint.interval == 0
        ):
            save_model(
                model=self.model,
                epoch=epoch + 1,
                save_dir=self.model_save_dir,
                model_name=self.model_name,
                use_date=False,
                save_as_checkpoint=True,
            )

    def __train_epoch(self, tqdm_loader: tqdm) -> float:
        """
        Process the training dataset for one epoch.

        Args:
            tqdm_loader (tqdm): Dataloader wrapped with a progress bar.

        Returns:
            float: Average training loss for the epoch.
        """
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(tqdm_loader):
            batch_loss, batch_size = self.__train_batch(batch, step)
            total_loss += batch_loss
            total_samples += batch_size

            # Update progress bar with the current average loss.
            avg_loss = total_loss / total_samples
            tqdm_loader.set_postfix(loss=f"{avg_loss:.4f}")

            if self.scheduler_step_per_batch:
                self.scheduler.step()

        return total_loss / total_samples

    def __train_batch(self, batch: Any, step: int) -> tuple[float, int]:
        """
        Perform a single training step on a batch.

        Args:
            batch (Any): A batch containing inputs and covariates.
            step (int): Current step index.

        Returns:
            Tuple[float, int]: Batch loss value and number of samples in the batch.
        """
        data, covariates = batch  # Unpack batch data.
        data = data.to(self.device)
        covariates = covariates.to(self.device)

        # Use autocast for mixed precision if enabled.
        with autocast(
            enabled=self.properties.train.mixed_precision, device_type=self.device
        ):
            model_outputs = self.model(data, covariates)
            loss = self.loss(
                model_outputs=model_outputs,
                x=data,
                current_epoch=self.epoch,
                covariates=covariates,
            )

        loss_value = loss.item()

        # Account for gradient accumulation.
        if self.properties.train.gradient_accumulation:
            loss = loss / self.properties.train.gradient_accumulation_steps

        # Backpropagate using mixed precision if enabled.
        if self.properties.train.mixed_precision:
            self.__backpropagate_with_mixed_precision(loss, step)
        else:
            self.__backpropagate(loss, step)

        return loss_value, data.size(0)

    def __backpropagate_with_mixed_precision(self, loss: Tensor, step: int) -> None:
        """
        Backpropagate the loss with AMP support and gradient accumulation.

        Args:
            loss (Tensor): The computed loss.
            step (int): Current step index.
        """
        self.scaler.scale(loss).backward()

        # Perform gradient clipping if enabled.
        if self.properties.train.gradient_clipping:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(), self.properties.train.gradient_clipping_value
            )

        if self.properties.train.gradient_accumulation:
            if (step + 1) % self.properties.train.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def __backpropagate(self, loss: Tensor, step: int) -> None:
        """
        Backpropagate the loss without AMP.

        Args:
            loss (Tensor): The computed loss.
            step (int): Current step index.
        """
        loss.backward()

        if self.properties.train.gradient_clipping:
            clip_grad_norm_(
                self.model.parameters(), self.properties.train.gradient_clipping_value
            )

        if self.properties.train.gradient_accumulation:
            if (step + 1) % self.properties.train.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def __validate(self) -> float:
        """
        Validate the model on the validation dataset.

        Returns:
            float: The average validation loss.
        """
        self.model.eval()  # Set model to evaluation mode.
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                data, covariates = batch
                data = data.to(self.device)
                covariates = covariates.to(self.device)
                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    model_outputs = self.model(data, covariates)
                    loss = self.loss(
                        model_outputs=model_outputs,
                        x=data,
                        current_epoch=self.epoch,
                        covariates=covariates,
                    )
                total_val_loss += loss.item()
                total_val_samples += data.size(0)

        return total_val_loss / total_val_samples

    def __report_enabled_optimizations(self) -> None:
        """Log the optimizations enabled in the training configuration."""
        if self.properties.train.cross_validation:
            self.logger.info(
                f"Enabled cross-validation with {self.properties.train.cross_validation_folds} folds."
            )
        if self.properties.train.mixed_precision:
            self.logger.info("Enabled mixed precision training.")
        if self.properties.train.gradient_accumulation:
            self.logger.info(
                f"Enabled gradient accumulation with steps: {self.properties.train.gradient_accumulation_steps}"
            )
        if self.properties.train.gradient_clipping:
            self.logger.info(
                f"Enabled gradient clipping with value: {self.properties.train.gradient_clipping_value}"
            )
        if self.properties.train.early_stopping.enabled:
            self.logger.info(
                f"Enabled early stopping with patience: {self.properties.train.early_stopping.patience}"
            )
        if self.properties.train.checkpoint.save_checkpoint:
            self.logger.info(
                f"Enabled model checkpointing every {self.properties.train.checkpoint.interval} epochs."
            )
        if self.properties.train.save_model:
            self.logger.info("Enabled saving the final model.")
