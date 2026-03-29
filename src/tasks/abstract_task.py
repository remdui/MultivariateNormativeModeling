"""Abstract Task class for all tasks."""

from abc import ABC, abstractmethod
from inspect import Parameter, Signature, signature
from logging import Logger
from typing import Any

import torch
from torch import Tensor

from data.factory import get_dataloader
from entities.experiment_manager import ExperimentManager
from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.abstract_model import AbstractModel
from model.models.factory import get_model
from optimization.loss_functions.factory import get_loss_function
from tasks.task_result import TaskResult
from util.system_utils import gpu_supported_by_triton_compiler


class AbstractTask(ABC):
    """
    Abstract Task class for all tasks.

    This class provides a template for tasks within the VAE pipeline by setting up
    common components such as data loaders, model initialization, and loss function
    configuration. Subclasses must implement the `run()` method to define task-specific
    behavior.
    """

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """
        Initialize the AbstractTask.

        Sets up the task by initializing data loaders, input/output dimensions,
        model, and loss function.
        """
        self.logger = logger
        self.properties = Properties.get_instance()
        self.device = self.properties.system.device
        self.model: AbstractModel  # To be defined in __build_model
        self.experiment_manager = ExperimentManager.get_instance()
        self.task_name = "unset"
        self.model_save_dir = self.properties.system.models_dir
        self.model_name = f"{self.properties.meta.name}_v{self.properties.meta.version}"
        self._loss_forward_signature: Signature | None = None
        self.__setup_task()

    @abstractmethod
    def run(self) -> TaskResult:
        """
        Execute the task.

        Subclasses must implement this method to define the main logic of the task.

        Returns:
            TaskResult: The result of executing the task.
        """

    def get_task_name(self) -> str:
        """
        Get the name of the task.

        Returns:
            str: The task name.
        """
        return self.task_name

    def __setup_task(self) -> None:
        """Set up all components of the task."""
        # Initialize data loader and dimensions
        self.__initialize_dataloader()
        self.__initialize_dimensions()

        # Build model and set up loss function
        self.__build_model()
        self.__setup_loss_function()

    def __initialize_dataloader(self) -> None:
        """Initialize the data loader and assign train, test, and (if applicable) validation loaders."""
        self.dataloader = get_dataloader(self.properties.dataset.data_type)
        self.train_dataloader = self.dataloader.train_dataloader()
        self.test_dataloader = self.dataloader.test_dataloader()

        if not self.properties.train.cross_validation:
            self.val_dataloader = self.dataloader.val_dataloader()

        self.logger.info(f"Initialized Dataloader: {self.dataloader}")

    def __initialize_dimensions(self) -> None:
        """
        Initialize input and output dimensions based on the training dataset.

        Assumes the first sample's data tensor defines the input dimension.
        """
        self.input_dim = self.train_dataloader.dataset[0][0].shape[0]
        self.output_dim = self.input_dim
        self.batch_size = self.properties.train.batch_size

    def __build_model(self) -> None:
        """Build and initialize the model based on the configuration."""
        # Get the model using the factory
        model = get_model(
            self.properties.model.architecture,
            self.input_dim,
            self.output_dim,
            len(self.dataloader.get_encoded_covariate_labels()),
        )
        model = model.to(self.device)

        if gpu_supported_by_triton_compiler():
            self.logger.info(
                "GPU is supported by the Triton compiler, enabling model compilation."
            )
            self.model = torch.compile(model)  # type: ignore
        else:
            self.model = model

        # Extract and log model configuration details
        activation = self.properties.model.activation_function
        activation_params = self.properties.model.activation_function_params.get(
            activation, {}
        )
        final_activation = self.properties.model.final_activation_function
        final_activation_params = self.properties.model.activation_function_params.get(
            final_activation, {}
        )
        normalization = self.properties.model.normalization_layer
        normalization_params = self.properties.model.normalization_layer_params.get(
            normalization, {}
        )

        self.logger.info(
            f"Activation function: {activation} with parameters {activation_params}"
        )
        self.logger.info(
            f"Final Activation function: {final_activation} with parameters {final_activation_params}"
        )
        self.logger.info(
            f"Normalization layer: {normalization} with parameters {normalization_params}"
        )
        self.logger.info(f"Initialized model: {self.model}")
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Model Parameters: {trainable_params}")

    def __setup_loss_function(self) -> None:
        """Configure the loss function based on the training configuration."""
        loss_function_name = self.properties.train.loss_function
        loss_function_params = self.properties.train.loss_function_params.get(
            loss_function_name, {}
        )

        self.logger.info(f"Base loss params from config: {loss_function_params}")
        self.loss = get_loss_function(loss_function_name, **loss_function_params)
        try:
            self._loss_forward_signature = signature(self.loss.forward)
        except (TypeError, ValueError):
            self._loss_forward_signature = None
        self.logger.info(
            f"Initialized loss function {self.loss} with parameters: {loss_function_params}"
        )

    def get_model(self) -> AbstractModel:
        """
        Retrieve the initialized model.

        Returns:
            AbstractModel: The model used for the task.
        """
        return self.model

    def get_input_size(self) -> int:
        """
        Get the input dimension size of the model.

        Returns:
            int: The size of the input dimension.
        """
        return self.input_dim

    def _compute_loss(
        self,
        model_outputs: dict[str, Tensor] | Tensor,
        x: Tensor,
        covariates: Tensor | None = None,
        current_epoch: int | None = None,
    ) -> Tensor:
        """
        Compute loss in a signature-aware way so both custom and stock PyTorch losses work.

        The method inspects the configured loss forward signature, forwards only accepted
        keyword arguments, and falls back to `(prediction, target)` positional calling.
        """
        recon_x = self.__extract_reconstruction_tensor(model_outputs)
        target = self.__resolve_reconstruction_target(recon_x, x, covariates)
        candidate_args = self.__build_loss_candidate_args(
            model_outputs=model_outputs,
            recon_x=recon_x,
            target=target,
            x=x,
            covariates=covariates,
            current_epoch=current_epoch,
        )

        forward_signature = self._loss_forward_signature
        if forward_signature is None:
            return self.loss(recon_x, target)

        accepted_kwargs = self.__select_loss_kwargs(forward_signature, candidate_args)
        required_param_names = self.__get_required_parameter_names(forward_signature)
        missing_required = [
            name for name in required_param_names if name not in accepted_kwargs
        ]
        has_positional_only_required = any(
            param.kind is Parameter.POSITIONAL_ONLY
            and param.default is Parameter.empty
            for name, param in forward_signature.parameters.items()
            if name != "self"
        )

        if not missing_required and not has_positional_only_required:
            return self.loss(**accepted_kwargs)

        required_positional_count = sum(
            1
            for name, param in forward_signature.parameters.items()
            if name != "self"
            and param.default is Parameter.empty
            and param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        )

        if required_positional_count <= 1:
            return self.loss(recon_x)
        return self.loss(recon_x, target)

    def __extract_reconstruction_tensor(
        self, model_outputs: dict[str, Tensor] | Tensor
    ) -> Tensor:
        """Extract the tensor used as model prediction for generic losses."""
        if isinstance(model_outputs, Tensor):
            return model_outputs
        if "x_recon" in model_outputs:
            return model_outputs["x_recon"]
        raise ValueError("Model outputs must contain 'x_recon' for loss computation.")

    def __resolve_reconstruction_target(
        self, recon_x: Tensor, x: Tensor, covariates: Tensor | None
    ) -> Tensor:
        """
        Resolve the reconstruction target tensor.

        For embedding modes that append covariates to decoder outputs, use [x, covariates].
        """
        if covariates is None or covariates.numel() == 0:
            return x
        if recon_x.ndim != x.ndim or recon_x.shape[:-1] != x.shape[:-1]:
            return x
        if recon_x.shape[-1] == x.shape[-1] + covariates.shape[-1]:
            return torch.cat([x, covariates], dim=-1)
        return x

    def __build_loss_candidate_args(
        self,
        model_outputs: dict[str, Tensor] | Tensor,
        recon_x: Tensor,
        target: Tensor,
        x: Tensor,
        covariates: Tensor | None,
        current_epoch: int | None,
    ) -> dict[str, Any]:
        """Build a superset of possible loss arguments for signature filtering."""
        return {
            "model_outputs": model_outputs,
            "x": x,
            "covariates": covariates,
            "covariate_labels": self.dataloader.get_encoded_covariate_labels(),
            "current_epoch": current_epoch,
            "input": recon_x,
            "input1": recon_x,
            "prediction": recon_x,
            "predictions": recon_x,
            "y_pred": recon_x,
            "recon_x": recon_x,
            "x_recon": recon_x,
            "target": target,
            "input2": target,
            "y_true": target,
            "labels": target,
        }

    def __select_loss_kwargs(
        self, forward_signature: Signature, candidate_args: dict[str, Any]
    ) -> dict[str, Any]:
        """Select only kwargs accepted by the configured loss forward signature."""
        accepts_var_kwargs = any(
            param.kind is Parameter.VAR_KEYWORD
            for param in forward_signature.parameters.values()
        )
        if accepts_var_kwargs:
            return candidate_args
        return {
            name: candidate_args[name]
            for name, param in forward_signature.parameters.items()
            if name != "self"
            and name in candidate_args
            and param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

    def __get_required_parameter_names(self, forward_signature: Signature) -> list[str]:
        """Return required non-self parameter names from a forward signature."""
        return [
            name
            for name, param in forward_signature.parameters.items()
            if name != "self"
            and param.default is Parameter.empty
            and param.kind
            in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
            )
        ]
