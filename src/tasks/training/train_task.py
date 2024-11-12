"""Train the model using the configuration."""

import torch
from torch import GradScaler, Tensor, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from entities.log_manager import LogManager
from model.layers.initialization.factory import initialize_weights
from optimization.optimizers.factory import get_optimizer
from optimization.schedulers.factory import get_scheduler
from tasks.abstract_task import AbstractTask
from tasks.task_result import TaskResult
from util.model_utils import save_model


class TrainTask(AbstractTask):
    """Trainer class to train the model."""

    def __init__(self) -> None:
        """Initialize the Trainer class."""
        super().__init__(LogManager.get_logger(__name__))
        self.logger.info("Initializing TrainTask.")
        self.__init_train_task()

    def __init_train_task(self) -> None:
        """Setup the train task."""
        self.task_name = "train"
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__setup_regularization()
        self.__setup_amp()
        self.__initialize_weights()
        self.__report_enabled_optimizations()

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

    def __setup_regularization(self) -> None:
        """TODO: Implement regularization setup."""
        self.logger.info("Initialized regularization: None")

    def __setup_amp(self) -> None:
        """Initialize automatic mixed precision (AMP) for training."""
        self.scaler = GradScaler()

    def __initialize_weights(self) -> None:
        """Weight initialization of the model.

        Weight initialization improves the convergence of the model during training.
        See this paper for more details: https://arxiv.org/abs/1704.08863
        He initialization is commonly used for ReLU activation functions, see https://arxiv.org/pdf/1502.01852
        """
        # Retrieve the weight initializer from the configuration
        weight_initializer = self.properties.model.weight_initializer

        # Initialize the model weights
        initialize_weights(self.model, weight_initializer)
        self.logger.info(f"Initialized weights using {weight_initializer}")

    def run(self) -> TaskResult:
        """Train the model."""
        epochs = self.properties.train.epochs
        current_learning_rate = self.scheduler.get_last_lr()[0]

        # Initialize the training result
        results = TaskResult()

        self.logger.info(
            f"Training model: {self.model_name} for {epochs} epochs with batch size {self.batch_size} using device {self.device}"
        )

        for epoch in range(epochs):
            self.model.train()

            tqdm_loader = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            avg_loss = self.__process_batch(tqdm_loader)

            self.logger.info(
                f"Finished epoch {epoch + 1}: average training loss: {avg_loss:.4f}"
            )

            # Validation loop
            avg_val_loss = self.__validate()
            self.logger.info(
                f"Average validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}"
            )

            if not self.scheduler_step_per_batch:
                if self.properties.train.scheduler == "plateau":
                    self.scheduler.step(avg_val_loss)  # type: ignore
                else:
                    self.scheduler.step()

                if current_learning_rate != self.scheduler.get_last_lr()[0]:
                    current_learning_rate = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        f"Scheduler adjusted the learning rate to: {self.scheduler.get_last_lr()[0]}"
                    )

            # Early stopping check
            if self.properties.train.early_stopping.enabled:
                if self.__early_stopping_check(avg_val_loss):
                    self.logger.info("Early stopping triggered.")
                    break

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

        # Save the final model
        if self.properties.train.save_model:
            save_model(
                model=self.model,
                save_dir=self.model_save_dir,
                model_name=self.model_name,
                use_date=False,
            )

        self.logger.info("Training completed.")
        return results

    def __process_batch(self, tqdm_loader: tqdm) -> float:
        """Process a batch of data."""
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm_loader:
            batch_loss, batch_size = self.__train_step(batch)
            total_loss += batch_loss  # Accumulate the loss
            total_samples += batch_size  # Accumulate the number of samples

            # Update the progress bar and report the average loss
            avg_loss = total_loss / total_samples
            tqdm_loader.set_postfix(loss=f"{avg_loss:.4f}")

            if self.scheduler_step_per_batch:
                self.scheduler.step()

        avg_loss = total_loss / total_samples

        return avg_loss

    def __train_step(self, batch: Tensor) -> tuple[float, int]:
        """Perform a single training step."""
        data, _ = batch  # Unpack batch (features, covariates)
        data = data.to(self.device)  # Move data to device

        self.optimizer.zero_grad()  # Zero gradients

        # Mixed precision training
        with autocast(
            enabled=self.properties.train.mixed_precision, device_type=self.device
        ):  # Automatic mixed precision
            recon_batch, z_mean, z_logvar = self.model(data)  # Forward pass
            loss = self.loss(recon_batch, data, z_mean, z_logvar)  # Compute loss

        if self.properties.train.mixed_precision:
            self.scaler.scale(loss).backward()  # Backward pass
            if self.properties.train.gradient_clipping:
                self.scaler.unscale_(
                    self.optimizer
                )  # Unscales the gradients, see: https://pytorch.org/docs/main/notes/amp_examples.html#gradient-clipping
                clip_grad_norm_(
                    self.model.parameters(),
                    self.properties.train.gradient_clipping_value,
                )
            self.scaler.step(self.optimizer)  # Update weights
            self.scaler.update()  # Update scaler
        else:
            loss.backward()  # Backward pass
            if self.properties.train.gradient_clipping:
                clip_grad_norm_(
                    self.model.parameters(),
                    self.properties.train.gradient_clipping_value,
                )
            self.optimizer.step()  # Update weights

        return loss.item(), data.size(0)

    def __validate(self) -> float:
        """Validate the model on the validation set."""
        self.model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

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

    def __early_stopping_check(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
            return False

        if (
            val_loss
            < self.best_val_loss - self.properties.train.early_stopping.min_delta
        ):
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        if self.no_improvement_epochs >= self.properties.train.early_stopping.patience:
            return True
        return False

    def __report_enabled_optimizations(self) -> None:
        """Report the enabled optimizations."""
        if self.properties.train.mixed_precision:
            self.logger.info("Enabled mixed precision training.")
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
