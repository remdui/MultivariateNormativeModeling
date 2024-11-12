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
    type: str
    params: dict[str, Any]


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


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""

    save_checkpoint: bool = True
    interval: int = 20
    use_checkpoint: bool = False
    checkpoint: str = ""


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
    image_type: str = "grayscale"
    internal_file_format: str = "hdf"
    pin_memory: bool = True
    shuffle: bool = True
    test_split: float = 0.1
    train_split: float = 0.8
    val_split: float = 0.2
    enable_transforms: bool = True
    transforms: list[TransformConfig] = [
        TransformConfig(
            name="NoiseTransform",
            type="preprocessing",
            params={"mean": 0.0, "std": 0.05, "distribution": "normal"},
        ),
        TransformConfig(
            name="DataCleaningTransform",
            type="preprocessing",
            params={"drop_na": True, "remove_duplicates": True},
        ),
        TransformConfig(
            name="NormalizationTransform",
            type="preprocessing",
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
            # "cnn_vae": {
            #     "encoder": "cnn",
            #     "decoder": "cnn",
            #     "latent_dim": 32,
            #     "conv_layers": {
            #         "num_layers": 2,
            #         "layer_config": [
            #             {
            #                 "out_channels": 32,
            #                 "kernel_size": 3,
            #                 "stride": 1,
            #                 "padding": 1,
            #                 "pool_type": "",
            #             },
            #             {
            #                 "out_channels": 64,
            #                 "kernel_size": 3,
            #                 "stride": 1,
            #                 "padding": 1,
            #                 "pool_type": "",
            #             },
            #         ],
            #     },
            # },
        }
    )

    # Model components
    hidden_layers: list[int] = [1024, 512, 256]
    weight_initializer: str = "he_normal"

    # Layer-specific configurations
    dropout: DropoutConfig = Field(default_factory=DropoutConfig)

    normalization_layer: str = "batchnorm1d"
    normalization_layer_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "batchnorm1d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
            },
            "batchnorm2d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
            },
            "batchnorm3d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
            },
            "groupnorm": {
                "eps": 1e-5,
                "affine": True,
            },
            "syncbatchnorm": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
            },
            "instancenorm1d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": False,
                "track_running_stats": False,
            },
            "instancenorm2d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": False,
                "track_running_stats": False,
            },
            "instancenorm3d": {
                "eps": 1e-5,
                "momentum": 0.1,
                "affine": False,
                "track_running_stats": False,
            },
            "layernorm": {
                "eps": 1e-5,
                "elementwise_affine": True,
            },
            "localresponsenorm": {
                "alpha": 1e-4,
                "beta": 0.75,
                "k": 1.0,
            },
            "rmsnorm": {
                "eps": 1e-8,
                "elementwise_affine": True,
            },
        }
    )

    activation_function: str = "relu"
    final_activation_function: str = "sigmoid"
    activation_function_params: dict[str, Any] = Field(
        default_factory=lambda: {
            # PyTorch activation functions (weighted sum, non-linearity)
            "elu": {
                "alpha": 1.0,
            },
            "hardshrink": {
                "lambd": 0.5,
            },
            "hardtanh": {
                "min_val": -1.0,
                "max_val": 1.0,
            },
            "leakyrelu": {
                "negative_slope": 0.01,
            },
            "prelu": {
                "num_parameters": 1,
                "init": 0.25,
            },
            "rrelu": {
                "lower": 0.125,
                "upper": 0.333,
                "inplace": False,
            },
            "celu": {
                "alpha": 1.0,
            },
            "gelu": {
                "approximate": "none",
            },
            "softplus": {
                "beta": 1.0,
                "threshold": 20.0,
            },
            "softshrink": {
                "lambd": 0.5,
            },
            "threshold": {
                "threshold": 1.0,
                "value": 0.0,
            },
            "glu": {
                "dim": -1,
            },
            # PyTorch activation functions (other)
            "adaptivelogsoftmaxwithloss": {
                "cutoffs": [10, 20, 30],
                "div_value": 4.0,
                "head_bias": False,
            },
            "logsoftmax": {
                "dim": None,
            },
            "softmax": {
                "dim": None,
            },
            "softmin": {
                "dim": None,
            },
        }
    )


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
    epochs: int = 20
    gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 10
    gradient_clipping: bool = True
    gradient_clipping_value: float = 1.0
    mixed_precision: bool = False
    save_model: bool = True
    save_format: str = "safetensors"

    # Grouped configurations
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    loss_function: str = "bce_vae"
    loss_function_params: dict[str, Any] = Field(
        default_factory=lambda: {
            # Custom loss function implementations
            "bce_vae": {
                "reduction": "sum",
            },
            "mse_vae": {
                "reduction": "sum",
            },
            # PyTorch loss functions
            "l1": {
                "reduction": "sum",
            },
            "mse": {
                "reduction": "sum",
            },
            "cross_entropy": {
                "ignore_index": -100,
                "reduction": "sum",
                "label_smoothing": 0.0,
            },
            "ctc": {
                "blank": 0,
                "reduction": "sum",
                "zero_infinity": False,
            },
            "nll": {
                "ignore_index": -100,
                "reduction": "sum",
            },
            "poisson_nll": {
                "log_input": True,
                "full": False,
                "eps": 1e-06,
                "reduction": "sum",
            },
            "gaussian_nll": {
                "full": False,
                "eps": 1e-06,
                "reduction": "sum",
            },
            "kldiv": {
                "reduction": "sum",
                "log_target": False,
            },
            "bce": {
                "reduction": "sum",
            },
            "bce_with_logits": {
                "reduction": "sum",
            },
            "margin_ranking": {
                "margin": 0.0,
                "reduction": "sum",
            },
            "hinge_embedding": {
                "margin": 1.0,
                "reduction": "sum",
            },
            "multi_label_margin": {
                "reduction": "sum",
            },
            "huber": {
                "reduction": "sum",
                "delta": 1.0,
            },
            "smooth_l1": {
                "reduction": "sum",
                "beta": 1.0,
            },
            "soft_margin": {
                "reduction": "sum",
            },
            "multi_label_soft_margin": {
                "reduction": "sum",
            },
            "cosine_embedding": {
                "margin": 0.0,
                "reduction": "sum",
            },
            "multi_margin": {
                "p": 1.0,
                "margin": 1.0,
                "reduction": "sum",
            },
            "triplet_margin": {
                "margin": 1.0,
                "p": 2.0,
                "eps": 1e-06,
                "swap": False,
                "reduction": "sum",
            },
            "triplet_margin_with_distance": {
                "margin": 1.0,
                "swap": False,
                "reduction": "sum",
            },
        }
    )

    scheduler: str = "step"
    scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "default": {
                "last_epoch": -1,
            },
            "lambda": {
                "lr_lambda": "lambda epoch: 0.95 ** epoch",
                "last_epoch": -1,
            },
            "multiplicative": {
                "lr_lambda": "lambda epoch: 0.95",
                "last_epoch": -1,
            },
            "step": {
                "step_size": 10,
                "gamma": 0.1,
                "last_epoch": -1,
            },
            "multistep": {
                "milestones": [30, 80],
                "gamma": 0.1,
                "last_epoch": -1,
            },
            "constant": {
                "factor": 0.3333333333333333,
                "total_iters": 5,
                "last_epoch": -1,
            },
            "linear": {
                "start_factor": 0.3333333333333333,
                "end_factor": 1.0,
                "total_iters": 5,
                "last_epoch": -1,
            },
            "exponential": {
                "gamma": 0.95,
                "last_epoch": -1,
            },
            "polynomial": {
                "total_iters": 5,
                "power": 1.0,
                "last_epoch": -1,
            },
            "cosineannealing": {
                "T_max": 10,
                "eta_min": 0.0,
                "last_epoch": -1,
            },
            "plateau": {
                "mode": "min",
                "factor": 0.1,
                "patience": 10,
                "threshold": 0.0001,
                "threshold_mode": "rel",
                "cooldown": 0,
                "min_lr": 0,
                "eps": 1e-08,
            },
            "cyclic": {
                "base_lr": 0.001,
                "max_lr": 0.01,
                "step_size_up": 2000,
                "step_size_down": None,
                "mode": "triangular",
                "gamma": 1.0,
                "scale_fn": None,
                "scale_mode": "cycle",
                "cycle_momentum": True,
                "base_momentum": 0.8,
                "max_momentum": 0.9,
                "last_epoch": -1,
            },
            "onecycle": {
                "max_lr": 0.1,
                "total_steps": None,
                "epochs": None,
                "steps_per_epoch": None,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
                "cycle_momentum": True,
                "base_momentum": 0.85,
                "max_momentum": 0.95,
                "div_factor": 25.0,
                "final_div_factor": 10000.0,
                "three_phase": False,
                "last_epoch": -1,
            },
            "cosineannealingwarmrestarts": {
                "T_0": 10,
                "T_mult": 2,
                "eta_min": 0.0,
                "last_epoch": -1,
            },
        }
    )

    optimizer: str = "adam"
    optimizer_params: dict[str, Any] = Field(
        default_factory=lambda: {
            # PyTorch optimizers supported for cuda
            "adam": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "adamw": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
            },
            "sgd": {
                "lr": 0.001,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
            },
            # PyTorch optimizers
            "adadelta": {
                "lr": 1.0,
                "rho": 0.9,
                "eps": 1e-06,
                "weight_decay": 0,
            },
            "adafactor": {
                "lr": 0.01,
                "beta2_decay": -0.8,
                "eps": [None, 0.001],
                "d": 1.0,
                "weight_decay": 0.0,
            },
            "adagrad": {
                "lr": 0.01,
                "lr_decay": 0,
                "weight_decay": 0,
                "initial_accumulator_value": 0,
                "eps": 1e-10,
            },
            "sparse_adam": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "adamax": {
                "lr": 0.002,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
            },
            "asgd": {
                "lr": 0.01,
                "lambd": 0.0001,
                "alpha": 0.75,
                "t0": 1000000.0,
                "weight_decay": 0,
            },
            "lbfgs": {
                "lr": 1,
                "max_iter": 20,
                "max_eval": None,
                "tolerance_grad": 1e-07,
                "tolerance_change": 1e-09,
                "history_size": 100,
                "line_search_fn": None,
            },
            "nadam": {
                "lr": 0.002,
                "betas": [0.9, 0.999],
                "eps": 1e-08,
                "weight_decay": 0,
                "momentum_decay": 0.004,
                "decoupled_weight_decay": False,
            },
            "radam": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
                "decoupled_weight_decay": False,
            },
            "rmsprop": {
                "lr": 0.01,
                "alpha": 0.99,
                "eps": 1e-08,
                "weight_decay": 0,
                "momentum": 0,
                "centered": False,
            },
            "rprop": {
                "lr": 0.01,
                "etas": [0.5, 1.2],
                "step_sizes": [1e-06, 50],
            },
        }
    )


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
