"""Abstract Task class for all tasks."""

from abc import ABC, abstractmethod
from logging import Logger

from data.factory import get_dataloader
from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.abstract_model import AbstractModel
from model.models.factory import get_model
from optimization.loss_functions.factory import get_loss_function
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

    def get_task_name(self) -> str:
        """Return the task name."""
        return self.task_name

    def __setup_task(self) -> None:
        """Setup the components of a general task."""
        # Task name
        self.task_name = "unset"

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
        self.dataloader = get_dataloader(self.properties.dataset.data_type)

        self.train_dataloader = self.dataloader.train_dataloader()
        self.test_dataloader = self.dataloader.test_dataloader()

        if not self.properties.train.cross_validation:
            self.val_dataloader = self.dataloader.val_dataloader()

        self.logger.info(f"Initialized Dataloader: {self.dataloader}")

    def __initialize_dimensions(self) -> None:
        """Initialize the input and output dimensions."""
        self.input_dim = self.train_dataloader.dataset[0][0].shape[0]
        self.output_dim = self.input_dim
        self.batch_size = self.properties.train.batch_size

    def __build_model(self) -> None:
        """Build the model based on the configuration."""
        # Get the model based on the configuration
        model = get_model(
            self.properties.model.architecture, self.input_dim, self.output_dim
        )
        self.model = model.to(self.device)

        # print model architecture and parameters
        self.logger.info(
            f"Activation function: {self.properties.model.activation_function} with parameters {self.properties.model.activation_function_params.get(self.model.properties.model.activation_function, {})}"
        )
        self.logger.info(
            f"Final Activation function: {self.properties.model.final_activation_function} with parameters {self.properties.model.activation_function_params.get(self.model.properties.model.final_activation_function, {})}"
        )
        self.logger.info(
            f"Normalization layer: {self.properties.model.normalization_layer} with parameters {self.properties.model.normalization_layer_params.get(self.model.properties.model.normalization_layer, {})}"
        )
        self.logger.info(f"Initialized model: {self.model}")
        self.logger.info(
            f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def __setup_loss_function(self) -> None:
        """Get the loss function based on the configuration."""
        # Retrieve the name of the loss function (e.g., "mse")
        loss_function_name = self.properties.train.loss_function

        # Retrieve the parameters specific to the selected loss function from loss_function_params
        loss_function_params = self.properties.train.loss_function_params.get(
            loss_function_name, {}
        )

        # Initialize the loss function with the unpacked loss function parameters
        self.loss = get_loss_function(loss_function_name, **loss_function_params)

        self.logger.info(
            f"Initialized loss function {self.loss} with parameters: {loss_function_params}"
        )

    def get_model(self) -> AbstractModel:
        """Return the trained model."""
        return self.model

    def get_input_size(self) -> int:
        """Return the input size of the model."""
        return self.input_dim
