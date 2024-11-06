"""Train the model using the configuration."""

import torch
from torch import Tensor
from tqdm import tqdm

from data.impl.factory import get_dataloader
from entities.log_manager import LogManager
from entities.properties import Properties
from model.components.factory import get_decoder, get_encoder
from model.loss.factory import get_loss_function
from model.models.vae_modular import VAE
from model.optimizers.factory import get_optimizer
from model.schedulers.factory import get_scheduler
from util.model_utils import save_model


class Trainer:
    """Trainer class to train the model."""

    def __init__(self) -> None:
        """Initialize the Trainer class."""
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.device = self.properties.system.device
        self.__setup()

    def __setup(self) -> None:
        """Setup the Trainer class."""

        # Model save info
        self.model_save_dir = self.properties.system.models_dir
        self.model_name = f"{self.properties.meta.name}_v{self.properties.meta.version}"

        # Initialize data loader
        self.__initialize_dataloader()

        # Get input and output dimensions
        self.input_dim = self.train_dataloader.dataset[0][0].shape[0]
        self.output_dim = self.input_dim
        self.batch_size = self.properties.train.batch_size

        # Build model
        self.__build_model()

        # Setup training components
        self.__setup_optimizer()
        self.__setup_scheduler()
        self.__setup_loss_function()
        self.__setup_regularization()

    def __initialize_dataloader(self) -> None:
        """Initialize the data loader."""
        # Initialize data loader
        dataloader = get_dataloader(self.properties.dataset.data_type)

        self.train_dataloader = dataloader.train_dataloader()
        self.val_dataloader = dataloader.val_dataloader()
        self.test_dataloader = dataloader.test_dataloader()

        self.logger.info(f"Initialized Dataloader: {dataloader}")

    def __build_model(self) -> None:
        """Build the model based on the configuration."""
        encoder = get_encoder(
            self.properties.model.encoder,
            self.input_dim,
            self.properties.model.hidden_dim,
            self.properties.model.latent_dim,
        ).to(self.device)

        decoder = get_decoder(
            self.properties.model.decoder,
            self.properties.model.latent_dim,
            self.properties.model.hidden_dim[::-1],
            self.output_dim,
        ).to(self.device)

        self.model = VAE(encoder, decoder).to(self.device)
        # print model architecture and parameters

        self.logger.info(f"Initialized model: {self.model}")
        self.logger.info(
            f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

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
                "cycliclr",
                "onecyclelr",
                "cosineannealingwarmrestarts",
            }
        )
        self.logger.info(f"Initialized scheduler: {self.scheduler}")

    def __setup_loss_function(self) -> None:
        """Get the loss function based on the configuration."""
        self.loss_function = get_loss_function(self.properties.train.loss_function)
        self.logger.info(f"Initialized loss function: {self.loss_function}")

    def __setup_regularization(self) -> None:
        """TODO: Implement regularization setup."""
        self.logger.info("Initialized regularization: None")

    def __train_step(self, data: Tensor) -> float:
        """Perform a single training step."""
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        # TODO: Maybe clip gradients
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        """Train the model."""
        epochs = self.properties.train.epochs

        self.logger.info(
            f"Training model: {self.model_name} for {epochs} epochs with batch size {self.batch_size} using device {self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            tqdm_loader = tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{epochs}",
            )

            for batch in tqdm_loader:
                data, _ = batch
                data = data.to(self.device)

                batch_loss = self.__train_step(data)
                total_loss += batch_loss
                total_samples += data.size(0)

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
            val_loss = self.validate()
            self.logger.info(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")

            # Early stopping check
            if self.properties.train.early_stopping.enabled:
                if self.__early_stopping_check(val_loss):
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

    def validate(self) -> float:
        """Validate the model on the validation set."""
        self.model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                data, _ = batch
                data = data.to(self.device)

                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
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

    def get_model(self) -> VAE:
        """Return the trained model."""
        return self.model

    def get_input_size(self) -> int:
        """Return the input size of the model."""
        return self.input_dim
