"""Configuration schema is defined in this module."""

from math import isclose

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    normalization: str = "min_max"
    num_covariates: int = 2
    processed_data_file: str = "freesurfer_dummy_output.csv"
    raw_data_file: str = "freesurfer_output.rds"
    shuffle: bool = True
    test_split: float = 0.1
    train_split: float = 0.7
    val_split: float = 0.2

    @model_validator(mode="after")
    def validate_config_dataset_splits(self) -> "DatasetConfig":
        """Validate that the dataset splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if isclose(total, 1.0):
            return self
        raise ValueError("Train, validation, and test splits must sum to 1.0 or less.")


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

    encoder: str = "mlp"
    decoder: str = "mlp"
    activation_function: str = "relu"
    beta: float = 1.0
    covariate_embedding: str = "input_embedding"
    dropout_rate: float = 0.5
    hidden_dim: list[int] = [128, 64, 32]
    kl_annealing: bool = True
    kl_annealing_steps: int = 1000
    latent_dim: int = 10
    normalization_layer: str = "batch_norm"
    reconstruction_loss_weight: float = 1.0
    save_model: bool = True
    save_model_interval: int = 10
    save_model_path: str = "model.pth"
    weight_initialization: str = "xavier"


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    gamma: float = 0.1
    scheduler: str = "steplr"
    step_size: int = 10


class SystemConfig(BaseModel):
    """System configuration."""

    checkpoint: str = ""
    checkpoint_interval: int = 5
    data_dir: str = "./data"
    device: str = "cpu"
    log_dir: str = "./logs"
    models_dir: str = "./models"
    num_workers: int = 4
    output_dir: str = "./output"


class TrainConfig(BaseModel):
    """Training configuration."""

    batch_size: int = 32
    early_stopping: bool = True
    early_stopping_metric: str = "loss"
    early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 10
    epochs: int = 100
    gradient_clipping: float = 5.0
    learning_rate: float = 0.001
    log_interval: int = 10
    loss_function: str = "bce_kld"
    lr_warmup_steps: int = 500
    mixed_precision: bool = False
    optimizer: str = "adam"
    weight_decay: float = 0.0001


class ConfigSchema(BaseModel):
    """Configuration schema (following pydantic model validation)."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
