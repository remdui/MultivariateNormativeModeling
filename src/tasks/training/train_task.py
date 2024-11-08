"""Train the model using the configuration."""

import torch
from torch import Tensor
from tqdm import tqdm

from entities.log_manager import LogManager
from model.optimizers.factory import get_optimizer
from model.schedulers.factory import get_scheduler
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
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__setup_regularization()

    def __setup_optimizer(self) -> None:
        """Get the optimizer based on the configuration."""
        optimizer_params: dict = {}
        self.optimizer = get_optimizer(
            self.properties.train.optimizer,
            self.model.parameters(),
            float(self.properties.train.scheduler.learning_rate),
            **optimizer_params,
        )
        self.logger.info(f"Initialized optimizer: {self.optimizer}")

    def __setup_scheduler(self) -> None:
        """Get the scheduler based on the configuration."""
        scheduler_params = {
            "step_size": self.properties.train.scheduler.step_size,
            "gamma": self.properties.train.scheduler.gamma,
        }
        self.scheduler = get_scheduler(
            self.properties.train.scheduler.method.lower(),
            self.optimizer,
            **scheduler_params,
        )
        # Determine if scheduler steps per batch
        self.scheduler_step_per_batch = (
            self.properties.train.scheduler.method.lower()
            in {
                "cyclic",
                "onecycle",
                "cosineannealingwarmrestarts",
            }
        )
        self.logger.info(f"Initialized scheduler: {self.scheduler}")

    def __setup_regularization(self) -> None:
        """TODO: Implement regularization setup."""
        self.logger.info("Initialized regularization: None")

    def get_task_name(self) -> str:
        """Return the task name.

        Returns:
            str: The task name.
        """
        return "train"

    def run(self) -> TaskResult:
        """Train the model."""
        epochs = self.properties.train.epochs

        # Initialize the training result
        results = TaskResult()

        self.logger.info(
            f"Training model: {self.model_name} for {epochs} epochs with batch size {self.batch_size} using device {self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            tqdm_loader = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
            )

            for batch in tqdm_loader:
                batch_loss = self.__train_step(batch)
                total_loss += batch_loss  # Batch loss is the summed loss over the batch
                total_samples += (
                    self.properties.train.batch_size
                )  # increment total samples by batch size for correct average

                avg_loss = total_loss / total_samples
                tqdm_loader.set_postfix(loss=f"{avg_loss:.4f}")

                if self.scheduler_step_per_batch:
                    self.scheduler.step()

            if not self.scheduler_step_per_batch:
                self.scheduler.step()

            avg_loss = total_loss / total_samples
            self.logger.info(
                f"Finished epoch {epoch + 1}: average training loss: {avg_loss:.4f}"
            )

            # Validation loop
            avg_val_loss = self.__validate()
            self.logger.info(
                f"Average validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}"
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

    def __train_step(self, batch: Tensor) -> float:
        """Perform a single training step."""
        data, _ = batch
        data = data.to(self.device)

        # Initialize gradients for the optimizer for this batch
        self.optimizer.zero_grad()

        # Forward pass
        recon_batch, z_mean, z_logvar = self.model(data)

        # Compute loss
        loss = self.loss(recon_batch, data, z_mean, z_logvar)

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()

    def __validate(self) -> float:
        """Validate the model on the validation set."""
        self.model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                data, _ = batch
                data = data.to(self.device)

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
