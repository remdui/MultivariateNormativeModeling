"""Custom exceptions for the application."""


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""


class UnsupportedFileFormatError(ValueError):
    """Exception raised when an unsupported file format is encountered."""


class UnsupportedNormalizationMethodError(ValueError):
    """Exception raised when an unsupported normalization method is provided."""


class NoDataToSaveError(Exception):
    """Exception raised when no data is available to save after conversion."""


class UnsupportedCovariateEmbeddingTechniqueError(Exception):
    """Exception raised when an unsupported covariate embedding technique is encountered."""


class DataRowMismatchError(Exception):
    """Raised when the number of rows in the skipped data does not match the dataset."""
