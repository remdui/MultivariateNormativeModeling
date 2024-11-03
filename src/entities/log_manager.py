# util/log_utils.py

"""Utility functions and classes for logging."""

import logging
import os
from logging.handlers import RotatingFileHandler

from entities.properties import Properties
from util.errors import ConfigurationError


class LogManager:
    """Manages logging configuration."""

    _logger_initialized = False

    @staticmethod
    def setup_logging() -> None:
        """Set up logging configuration based on properties."""
        if LogManager._logger_initialized:
            return  # Prevent reconfiguration

        try:
            properties = Properties.get_instance()
            log_dir = properties.system.log_dir
            debug = properties.general.debug
            verbose = properties.general.verbose
            log_level_str = properties.general.log_level.upper()

            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "application.log")

            # Determine the log level
            if verbose:
                log_level = logging.NOTSET  # Capture all messages
            else:
                log_level = getattr(logging, log_level_str, logging.INFO)

            # Create formatters
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
            )

            # Create handlers
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setLevel(logging.NOTSET if verbose else log_level)
            file_handler.setFormatter(formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            if verbose:
                console_handler.setLevel(logging.NOTSET)
            elif debug:
                console_handler.setLevel(log_level)
            else:
                console_handler.setLevel(
                    logging.CRITICAL + 1
                )  # Effectively disables console output

            console_handler.setFormatter(formatter)

            # Get the root logger
            root_logger = logging.getLogger()
            # Remove any existing handlers
            root_logger.handlers = []
            # Add the new handlers
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            # Set the root logger level
            root_logger.setLevel(logging.NOTSET)

            LogManager._logger_initialized = True

        except ConfigurationError:
            # If Properties is not initialized yet or any other exception occurs,
            # keep using the temporary logger
            logging.getLogger(__name__).warning(
                "LoggerManager setup failed; using temporary logger."
            )
            LogManager._logger_initialized = (
                True  # Prevent further attempts until reconfiguration
            )

    @staticmethod
    def reconfigure_logging() -> None:
        """Reconfigure logging after Properties are initialized."""
        # Reset the initialization flag to allow reconfiguration
        LogManager._logger_initialized = False
        LogManager.setup_logging()

    @staticmethod
    def get_logger(name: str = "") -> logging.Logger:
        """Get a logger with the specified name to identify the source of the log message.

        Args:
            name (Optional[str], optional): Name of the logger. Defaults to "" (root logger).

        Returns:
            logging.Logger: Configured logger instance.
        """
        LogManager.setup_logging()
        return logging.getLogger(name)
