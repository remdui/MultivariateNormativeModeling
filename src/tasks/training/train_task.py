"""Train the model using the configuration."""

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
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.file_utils import write_results_to_file
from util.model_utils import save_model, visualize_model_arch


class TrainTask(AbstractTask):
    """Trainer class to train the model."""

    def __init__(self, trial: optuna.Trial | None = None) -> None:
        """Initialize the Trainer class."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TrainTask.")
        self.__init_train_task()
        self.trial = trial
        self.experiment_manager.clear_output_directory()

    def __init_train_task(self) -> None:
        """Setup the train task."""
        self.task_name = "train"
        self.experiment_manager.create_new_experiment(self.task_name)
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__setup_amp()
        self.__initialize_weights()
        self.__initialize_early_stopping()
        self.__report_enabled_optimizations()

    def __reinitialize_train_task(self) -> None:
        """Reinitialize the training task."""
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__initialize_weights()
        self.early_stopping.reset()

    def __setup_optimizer(self) -> None:
        """Get the optimizer based on the configuration."""
        # Retrieve the name of the optimizer (e.g., "adam")
        optimizer_name = self.properties.train.optimizer

        # Retrieve the parameters specific to the selected optimizer from optimizer_params
        optimizer_params = self.properties.train.optimizer_params.get(
            optimizer_name, {}
        )

        # Initialize the optimizer with model parameters and the unpacked optimizer parameters
        self.optimizer = get_optimizer(
            optimizer_name, self.model.parameters(), **optimizer_params
        )
        self.logger.info(
            f"Initialized optimizer {optimizer_name} with parameters: {self.optimizer.state_dict()}"
        )

    def __setup_scheduler(self) -> None:
        """Get the scheduler based on the configuration."""
        # Retrieve the scheduler method (e.g., "step")
        scheduler_method = self.properties.train.scheduler

        # Retrieve the parameters specific to the selected scheduler from scheduler_params
        scheduler_params = self.properties.train.scheduler_params.get(
            scheduler_method, {}
        )

        # TODO: Better integrate these settings for onecycle and cyclic schedulers
        # scheduler_params["steps_per_epoch"] = len(self.train_dataloader)
        # scheduler_params["epochs"] = self.properties.train.epochs

        self.scheduler = get_scheduler(
            self.properties.train.scheduler,
            self.optimizer,
            **scheduler_params,
        )

        # Determine if scheduler steps per batch
        self.scheduler_step_per_batch = self.properties.train.scheduler in {
            "cyclic",
            "onecycle",
            "cosineannealingwarmrestarts",
        }

        self.logger.info(
            f"Initialized scheduler {scheduler_method} with parameters: {self.scheduler.state_dict()}"  # type: ignore
        )
        self.logger.info(f"Scheduler steps per batch: {self.scheduler_step_per_batch}")

    def __setup_amp(self) -> None:
        """Initialize automatic mixed precision (AMP) for training."""
        self.scaler = GradScaler()

    def __initialize_weights(self) -> None:
        """Weight initialization of the model.

        Weight initialization improves the convergence of the model during training.
        See this paper for more details: https://arxiv.org/abs/1704.08863
        He initialization is commonly used for ReLU activation functions, see https://arxiv.org/pdf/1502.01852
        """
        # Reset the model parameters
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # Retrieve the weight initializer from the configuration
        weight_initializer = self.properties.model.weight_initializer

        # Initialize the model weights
        initialize_weights(self.model, weight_initializer)

        self.logger.info(f"Initialized weights using {weight_initializer}")

    def __initialize_early_stopping(self) -> None:
        """Initialize early stopping."""
        self.early_stopping = EarlyStopping()

    def run(self) -> TaskResult:
        """Train the model."""
        # Initialize the training result
        results = TaskResult()
        epochs = self.properties.train.epochs

        self.logger.info(
            f"Training model: {self.model_name} for {epochs} epochs with batch size {self.batch_size} using device {self.device}"
        )

        if self.properties.train.cross_validation:
            self.__run_cross_validation_training(epochs, results)
        else:
            self.__run_training(epochs, results)

        # Save the final model
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
        return results

    def __run_cross_validation_training(
        self, epochs: int, results: TaskResult
    ) -> TaskResult:
        """Run cross-validation training.

        Args:
            epochs (int): Number of epochs to train.
            results (TaskResult): TaskResult object to store the results.

        Returns:
            TaskResult: TaskResult object with the training results.
        """
        for fold in range(self.properties.train.cross_validation_folds):
            self.logger.info(
                f"Starting fold {fold + 1} of {self.properties.train.cross_validation_folds}"
            )

            # Get the training and validation dataloaders for the current fold
            train_dataloader, val_dataloader = self.dataloader.fold_dataloader(fold)

            # Overwrite the dataloaders for the current fold
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

            # Reinitialize the training task if not the first fold
            if fold > 0:
                self.__reinitialize_train_task()

            # Run the training loop
            self.__run_training(epochs, results, fold + 1)

        return results

    def __run_training(
        self, epochs: int, results: TaskResult, fold: int | None = None
    ) -> TaskResult:
        """Run the training loop.

        Args:
            epochs (int): Number of epochs to train.
            results (TaskResult): TaskResult object to store the results.
            fold (int | None): The fold number if using cross-validation, None otherwise.

        Returns:
            TaskResult: TaskResult object with the training results.
        """
        current_learning_rate = self.scheduler.get_last_lr()[0]
        best_val_loss = float("inf")  # Initialize with infinity

        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()  # Set model to training mode

            # Process the training data for the current epoch
            avg_loss = self.__process_training_epoch(epoch, epochs)
            avg_val_loss = self.__process_validation_epoch(epoch)

            # If using Optuna, report the intermediate result
            if self.trial is not None:
                if epoch > 100:  # start reporting after warm-up
                    self.trial.report(avg_val_loss, step=epoch)
                    # If the trial should be pruned, raise an exception
                    if self.trial.should_prune():
                        self.logger.info(
                            f"Trial {self.trial.number} pruned at epoch {epoch + 1} with val_loss={avg_val_loss:.4f}"
                        )
                        raise optuna.TrialPruned()

            # Update best model if validation loss improves
            best_val_loss = self.__update_best_model(avg_val_loss, best_val_loss)

            # Store the results in the TaskResult
            self.__store_results(results, fold, avg_loss, avg_val_loss)

            # Step the scheduler if not using per-batch steps
            current_learning_rate = self.__step_scheduler(
                current_learning_rate, avg_val_loss
            )

            # Early stopping check
            if self.__check_early_stopping(avg_val_loss):
                break

            # Save model checkpoint if checkpointing is enabled and interval is reached
            self.__save_checkpoint(epoch)

        return results

    def __process_training_epoch(self, epoch: int, epochs: int) -> float:
        """Process the training data for the current epoch."""
        tqdm_loader = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        avg_loss = self.__train_epoch(tqdm_loader)
        self.logger.info(
            f"Average training loss after epoch {epoch + 1}: {avg_loss:.4f}"
        )
        return avg_loss

    def __process_validation_epoch(self, epoch: int) -> float:
        """Process the validation data for the current epoch."""
        avg_val_loss = self.__validate()
        self.logger.info(
            f"Average validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}"
        )
        return avg_val_loss

    def __update_best_model(self, avg_val_loss: float, best_val_loss: float) -> float:
        """Update the best model if validation loss improves."""
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
        """Store the results in the TaskResult."""
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
        """Step the scheduler if not using per-batch steps."""
        if not self.scheduler_step_per_batch:
            if self.properties.train.scheduler == "plateau":
                self.scheduler.step(avg_val_loss)  # type: ignore
            else:
                self.scheduler.step()

            # Update and log the learning rate if it has changed
            if current_learning_rate != self.scheduler.get_last_lr()[0]:
                current_learning_rate = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Scheduler adjusted the learning rate to: {self.scheduler.get_last_lr()[0]}"
                )
        return current_learning_rate

    def __check_early_stopping(self, avg_val_loss: float) -> bool:
        """Check if early stopping condition is met."""
        if self.properties.train.early_stopping.enabled:
            if self.early_stopping.stop_condition_met(avg_val_loss):
                self.logger.info("Early stopping triggered.")
                return True
        return False

    def __save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint if checkpointing is enabled and interval is reached."""
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
        """Process a batch of data."""
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(tqdm_loader):
            batch_loss, batch_size = self.__train_batch(batch, step)
            total_loss += batch_loss  # Accumulate the loss
            total_samples += batch_size  # Accumulate the number of samples

            # Update the progress bar and report the average loss
            avg_loss = total_loss / total_samples
            tqdm_loader.set_postfix(loss=f"{avg_loss:.4f}")

            if self.scheduler_step_per_batch:
                self.scheduler.step()

        avg_loss = total_loss / total_samples

        return avg_loss

    def __train_batch(self, batch: Any, step: int) -> tuple[float, int]:
        """Perform a single training step."""
        data, _ = batch  # Unpack batch (features, covariates)
        data = data.to(self.device)  # Move data to device

        # Mixed precision training support with autocast
        with autocast(
            enabled=self.properties.train.mixed_precision, device_type=self.device
        ):
            recon_batch, z_mean, z_logvar = self.model(data)  # Forward pass
            loss = self.loss(
                recon_batch, data, z_mean, z_logvar, current_epoch=self.epoch
            )  # Compute loss

        # Store original loss value before further processing
        loss_value = loss.item()

        # Account for gradient accumulation by dividing the loss by the accumulation steps
        if self.properties.train.gradient_accumulation:
            loss = loss / self.properties.train.gradient_accumulation_steps

        # Backpropagation with mixed precision support
        if self.properties.train.mixed_precision:
            self.__backpropagate_with_mixed_precision(loss, step)
        # Normal backpropagation without mixed precision
        else:
            self.__backpropagate(loss, step)

        return loss_value, data.size(0)

    def __backpropagate_with_mixed_precision(self, loss: Tensor, step: int) -> None:
        """Backpropagate the loss through the model with mixed precision support.

        Args:
            loss (Tensor): The loss value.
            step (int): The current step
        """
        self.scaler.scale(loss).backward()  # type: ignore

        # Gradient clipping
        if self.properties.train.gradient_clipping:
            # Unscales the gradients, see: https://pytorch.org/docs/main/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(),
                self.properties.train.gradient_clipping_value,
            )

        # Account for gradient accumulation
        if self.properties.train.gradient_accumulation:
            if (step + 1) % self.properties.train.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_dataloader):
                self.scaler.step(self.optimizer)  # Update weights
                self.scaler.update()  # Update scaler
                self.optimizer.zero_grad()  # Reset gradients
        else:
            self.scaler.step(self.optimizer)  # Update weights
            self.scaler.update()  # Update scaler
            self.optimizer.zero_grad()  # Reset gradients

    def __backpropagate(self, loss: Tensor, step: int) -> None:
        """Backpropagate the loss through the model.

        Args:
            loss (Tensor): The loss value.
            step (int): The current step
        """
        loss.backward()  # type: ignore

        # Gradient clipping
        if self.properties.train.gradient_clipping:
            clip_grad_norm_(
                self.model.parameters(),
                self.properties.train.gradient_clipping_value,
            )

        # Account for gradient accumulation
        if self.properties.train.gradient_accumulation:
            if (step + 1) % self.properties.train.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_dataloader):
                self.optimizer.step()  # Update weights
                self.optimizer.zero_grad()  # Reset gradients
        else:
            self.optimizer.step()  # Update weights
            self.optimizer.zero_grad()  # Reset gradients

    def __validate(self) -> float:
        """Validate the model on the validation set."""
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        total_val_samples = 0

        # Disable gradient computation in validation
        with torch.no_grad():
            for batch in self.val_dataloader:
                data, _ = batch
                data = data.to(self.device)

                # Use autocast in validation for mixed precision
                with autocast(
                    enabled=self.properties.train.mixed_precision,
                    device_type=self.device,
                ):
                    recon_batch, z_mean, z_logvar = self.model(data)
                    loss = self.loss(recon_batch, data, z_mean, z_logvar)

                total_val_loss += loss.item()
                total_val_samples += data.size(0)

        avg_val_loss = total_val_loss / total_val_samples
        return avg_val_loss

    def __report_enabled_optimizations(self) -> None:
        """Report the enabled optimizations."""
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
