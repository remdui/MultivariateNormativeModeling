"""Train the model using the configuration."""

import torch
from torch import Tensor
from tqdm import tqdm

from entities.log_manager import LogManager
from entities.properties import Properties
from model.components.factory import get_decoder, get_encoder
from model.loss.factory import get_loss_function
from model.models.vae_modular import VAE
from model.optimizers.factory import get_optimizer
from model.schedulers.factory import get_scheduler
from preprocessing.dataloader import FreeSurferDataloader
from util.model_utils import save_model


class Trainer:
    """Trainer class to train the model."""

    def __init__(self) -> None:
        """Initialize the Trainer class."""
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.device = self.properties.system.device
        self._setup()

    def _setup(self) -> None:
        """Setup the Trainer class."""

        # Model save info
        self.model_save_dir = self.properties.system.models_dir
        self.model_name = f"{self.properties.meta.name}_v{self.properties.meta.version}"

        # Initialize data loader
        self._initialize_dataloader()

        # Build model
        self._build_model()

        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_regularization()

    def _initialize_dataloader(self) -> None:
        """Initialize the data loader."""
        # Initialize data loader
        self.dataloader = FreeSurferDataloader.init_dataloader()
        self.input_dim = self.dataloader.dataset[0][0].shape[0]
        self.output_dim = self.input_dim  # Assuming reconstruction

    def _build_model(self) -> None:
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

    def _setup_optimizer(self) -> None:
        """Get the optimizer based on the configuration."""
        optimizer_params: dict = {}
        self.optimizer = get_optimizer(
            self.properties.train.optimizer,
            self.model.parameters(),
            float(self.properties.train.learning_rate),
            **optimizer_params,
        )

    def _setup_scheduler(self) -> None:
        """Get the scheduler based on the configuration."""
        scheduler_params = {
            "step_size": self.properties.scheduler.step_size,
            "gamma": self.properties.scheduler.gamma,
        }
        self.scheduler = get_scheduler(
            self.properties.scheduler.scheduler.lower(),
            self.optimizer,
            **scheduler_params,
        )
        # Determine if scheduler steps per batch
        self.scheduler_step_per_batch = self.properties.scheduler.scheduler.lower() in {
            "cycliclr",
            "onecyclelr",
            "cosineannealingwarmrestarts",
        }

    def _setup_loss_function(self) -> None:
        """Get the loss function based on the configuration."""
        self.loss_function = get_loss_function(self.properties.train.loss_function)

    def _setup_regularization(self) -> None:
        """TODO: Implement regularization setup."""

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a checkpoint of the model."""
        save_model(
            self.model,
            epoch + 1,
            self.model_save_dir + "/checkpoints",
            self.model_name,
            use_date=False,
        )

    def _save_model(self, epoch: int) -> None:
        """Save the final model."""
        save_model(
            self.model,
            epoch + 1,
            self.model_save_dir,
            self.model_name,
            use_date=False,
        )

    def _train_step(self, data: Tensor) -> float:
        """Perform a single training step."""
        data = data.float().to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        if self.properties.train.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.properties.train.gradient_clipping
            )

        self.optimizer.step()
        train_loss = loss.item()

        if self.scheduler_step_per_batch:
            self.scheduler.step()

        return train_loss

    def train(self) -> None:
        """Train the model."""
        epochs = self.properties.train.epochs

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            tqdm_loader = tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"Epoch {epoch+1}/{epochs}",
            )

            for _, (data, _) in tqdm_loader:
                train_loss_batch = self._train_step(data)
                train_loss += train_loss_batch

            if not self.scheduler_step_per_batch:
                self.scheduler.step()

            avg_loss = train_loss / float(len(self.dataloader.dataset))
            self.logger.info(f"Finished epoch {epoch+1}: loss: {avg_loss:.2f}")

            if (
                self.properties.model.save_model
                and (epoch + 1) % self.properties.model.save_model_interval == 0
            ):
                self._save_checkpoint(epoch + 1)

        # Save the final model
        self._save_model(epochs + 1)
        self.logger.info("Training completed.")
