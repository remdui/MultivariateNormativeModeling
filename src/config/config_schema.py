"""Configuration schema is defined in this module."""

from math import isclose
from typing import Any

from pydantic import BaseModel, Field, model_validator


###################################################################
# Sub-configurations used in the main configuration schema
###################################################################
class TransformConfig(BaseModel):
    """Transform configuration type."""

    name: str
    params: dict[str, Any]


class BatchNormConfig(BaseModel):
    """Batch normalization configuration."""

    enabled: bool = True
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


class LinearConfig(BaseModel):
    """Linear layer configuration."""

    bias: bool = True


class DropoutConfig(BaseModel):
    """Dropout configuration."""

    enabled: bool = True
    p: float = 0.2
    inplace: bool = False


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = True
    metric: str = "loss"
    min_delta: float = 0.0
    patience: int = 10


class SchedulerConfig(BaseModel):
    """Learning rate configuration."""

    learning_rate: float = 0.001
    warmup_steps: int = 500
    method: str = "step"
    gamma: float = 0.1
    step_size: int = 10
    kl_annealing: bool = True
    kl_annealing_steps: int = 1000


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""

    save_checkpoint: bool = True
    interval: int = 20
    use_checkpoint: bool = False
    checkpoint: str = ""


class RegularizationConfig(BaseModel):
    """Regularization configuration."""

    gradient_clipping: float = 5.0
    weight_decay: float = 0.0001


class ImageConfig(BaseModel):
    """Configuration for image data representation."""

    length: int = 28
    width: int = 28
    num_visual_samples: int = 5
    save_image_samples: bool = True
    show_image_samples: bool = False


###################################################################
# Main configuration schema
###################################################################
class DatasetConfig(BaseModel):
    """Dataset configuration."""

    covariates: list[str] = ["age", "sex"]
    input_data: str = "generated_data.rds"
    data_type: str = "tabular"
    internal_file_format: str = "hdf"
    shuffle: bool = True
    test_split: float = 0.1
    train_split: float = 0.8
    val_split: float = 0.2
    enable_transforms: bool = True
    transforms: list[TransformConfig] = [
        TransformConfig(
            name="DataCleaningTransform",
            params={"drop_na": True, "remove_duplicates": True},
        ),
        TransformConfig(
            name="NormalizationTransform",
            params={"method": "min-max"},
        ),
    ]

    @model_validator(mode="after")
    def validate_config_dataset_splits(self) -> "DatasetConfig":
        """Validate that the train and val dataset splits sum to 1.0."""
        total = self.train_split + self.val_split
        if isclose(total, 1.0):
            return self
        raise ValueError("Train and validation splits must sum to 1.0.")


class GeneralConfig(BaseModel):
    """General configuration."""

    debug: bool = False
    log_level: str = "INFO"
    seed: int = 42
    verbose: bool = False


class MetaConfig(BaseModel):
    """Metadata configuration."""

    config_version: int = 2
    description: str = "Variational Autoencoder design experiment setup"
    name: str = "vae_basic"
    version: int = 1


class ModelConfig(BaseModel):
    """Model configuration."""

    # Dynamic model components
    architecture: str = "vae"
    components: dict[str, Any] = Field(
        default_factory=lambda: {
            "vae": {
                "encoder": "mlp",
                "decoder": "mlp",
                "latent_dim": 32,
                "covariate_embedding": "input_embedding",
            },
        }
    )

    # Model components
    hidden_layers: list[int] = [1024, 512, 256]
    activation_function: str = "relu"
    final_activation_function: str = "sigmoid"
    normalization_layer: str = "batch_norm"
    weight_initialization: str = "xavier"

    # Layer-specific configurations
    batch_norm: BatchNormConfig = Field(default_factory=BatchNormConfig)
    linear: LinearConfig = Field(default_factory=LinearConfig)
    dropout: DropoutConfig = Field(default_factory=DropoutConfig)


class SystemConfig(BaseModel):
    """System configuration."""

    data_dir: str = "./data"
    device: str = "cpu"
    log_dir: str = "./logs"
    models_dir: str = "./models"
    num_workers: int = 4
    output_dir: str = "./output"


class TrainConfig(BaseModel):
    """Training configuration."""

    # General training settings
    batch_size: int = 32
    epochs: int = 100
    loss_function: str = "bce_vae"
    optimizer: str = "adam"
    mixed_precision: bool = False
    save_model: bool = True

    # Grouped configurations
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    regularization: RegularizationConfig = Field(default_factory=RegularizationConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class ValidationConfig(BaseModel):
    """Validation configuration."""

    model: str = ""
    baseline_model: str | None = None
    data_representation: str = "tabular"
    metrics: list[str] = ["mse", "mae"]
    image: ImageConfig = Field(default_factory=ImageConfig)


###################################################################
# Configuration schema definition
###################################################################
class ConfigSchema(BaseModel):
    """Configuration schema (following pydantic model validation)."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
