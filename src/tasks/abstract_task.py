"""Abstract Task class for all tasks."""

from abc import ABC, abstractmethod
from logging import Logger

from data.impl.factory import get_dataloader
from entities.log_manager import LogManager
from entities.properties import Properties
from model.components.factory import get_decoder, get_encoder
from model.loss.factory import get_loss_function
from model.models.abstract_model import AbstractModel
from model.models.impl.vae_modular import VAE
from tasks.task_result import TaskResult


class AbstractTask(ABC):
    """Abstract Task class for all tasks."""

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """Initialize the Task class."""
        # Define task attributes
        self.logger = logger
        self.properties = Properties.get_instance()
        self.device = self.properties.system.device
        self.model: AbstractModel

        # Set up the task
        self.__setup_task()

    @abstractmethod
    def run(self) -> TaskResult:
        """Run the task."""
        raise NotImplementedError("Method run() must be implemented in subclass.")

    def __setup_task(self) -> None:
        """Setup the components of a general task."""

        # Model save info
        self.model_save_dir = self.properties.system.models_dir
        self.model_name = f"{self.properties.meta.name}_v{self.properties.meta.version}"

        # Initialize data loader
        self.__initialize_dataloader()

        # Setup input and output dimensions
        self.__initialize_dimensions()

        # Build model
        self.__build_model()

        # Setup loss function
        self.__setup_loss_function()

    def __initialize_dataloader(self) -> None:
        """Initialize the data loader."""
        dataloader = get_dataloader(self.properties.dataset.data_type)

        self.train_dataloader = dataloader.train_dataloader()
        self.val_dataloader = dataloader.val_dataloader()
        self.test_dataloader = dataloader.test_dataloader()

        self.logger.info(f"Initialized Dataloader: {dataloader}")

    def __initialize_dimensions(self) -> None:
        """Initialize the input and output dimensions."""
        self.input_dim = self.train_dataloader.dataset[0][0].shape[0]
        self.output_dim = self.input_dim
        self.batch_size = self.properties.train.batch_size

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
        # TODO: make factory and config option for model type of AbstractModel
        self.model = VAE(encoder, decoder).to(self.device)
        # print model architecture and parameters

        self.logger.info(f"Initialized model: {self.model}")
        self.logger.info(
            f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def __setup_loss_function(self) -> None:
        """Get the loss function based on the configuration."""
        self.loss_function = get_loss_function(self.properties.train.loss_function)
        self.logger.info(f"Initialized loss function: {self.loss_function}")

    def get_model(self) -> AbstractModel:
        """Return the trained model."""
        return self.model

    def get_input_size(self) -> int:
        """Return the input size of the model."""
        return self.input_dim
