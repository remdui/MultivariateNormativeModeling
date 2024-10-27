"""Configuration schema is defined in this module."""


class ConfigSchema:
    """Configuration schema."""

    class Dataset:
        """Dataset configuration."""

        normalization = "min_max"
        num_covariates = 2
        processed_data_file = "freesurfer_dummy_output.csv"
        raw_data_file = "freesurfer_output.rds"
        shuffle = True
        test_split = 0.1
        train_split = 0.7
        val_split = 0.2

    class General:
        """General configuration."""

        debug = False
        log_level = "INFO"
        seed = 42
        verbose = False

    class Meta:
        """Metadata configuration."""

        config_version = 2
        description = "Variational Autoencoder design experiment setup"
        name = "vae_basic"
        model_version = 1

    class Model:
        """Model configuration."""

        activation_function = "relu"
        beta = 1.0
        covariate_embedding = "input_embedding"
        dropout_rate = 0.5
        hidden_dim = [128, 64, 32]
        kl_annealing = True
        kl_annealing_steps = 1000
        latent_dim = 10
        normalization_layer = "batch_norm"
        reconstruction_loss_weight = 1.0
        save_model = True
        save_model_interval = 10
        save_model_path = "model.pth"
        weight_initialization = "xavier"

    class Scheduler:
        """Scheduler configuration."""

        gamma = 0.1
        scheduler = "StepLR"
        step_size = 10

    class System:
        """System configuration."""

        checkpoint = ""
        checkpoint_interval = 5
        data_dir = "./data"
        device = "cpu"
        log_dir = "./logs"
        model_dir = "./models"
        num_workers = 4
        output_dir = "./output"

    class Train:
        """Training configuration."""

        batch_size = 32
        early_stopping = True
        early_stopping_metric = "loss"
        early_stopping_patience = 10
        epochs = 100
        gradient_clipping = 5.0
        learning_rate = 0.001
        log_interval = 10
        loss_function = "mse"
        lr_warmup_steps = 500
        mixed_precision = False
        optimizer = "adam"
        weight_decay = 0.0001
