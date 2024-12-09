# util/log_utils.py

"""Utility functions and classes for logging."""

import logging
import os
from logging import Handler
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

            # Define the log file
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
            file_handler = LogManager.__create_file_handler(
                formatter, log_file, log_level, verbose
            )
            console_handler = LogManager.__create_console_handler(
                debug, formatter, log_level, verbose
            )

            # Configure the root logger with the handlers
            root_logger = logging.getLogger()
            root_logger.handlers = []
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.NOTSET)

            # Mark the logging as initialized
            LogManager._logger_initialized = True

            logger = LogManager.get_logger(__name__)

            logger.info(
                f"Logging initialized with settings: verbose={verbose}, debug={debug}, log_level={log_level_str}"
            )

        except ConfigurationError:
            # If Properties is not initialized yet or any other exception occurs,
            # keep using the temporary logger
            logging.getLogger(__name__).warning(
                "Properties have not been initialized yet. Temporarily using basic logging configuration."
            )
            LogManager._logger_initialized = (
                True  # Prevent further attempts until reconfiguration
            )

    @staticmethod
    def __create_file_handler(
        formatter: logging.Formatter, log_file: str, log_level: int, verbose: bool
    ) -> Handler:
        """Create a file handler."""
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=100
        )
        file_handler.setLevel(logging.NOTSET if verbose else log_level)
        file_handler.setFormatter(formatter)

        # If the log file already exists, rotate it right now
        if os.path.exists(log_file):
            file_handler.doRollover()

        return file_handler

    @staticmethod
    def __create_console_handler(
        debug: bool, formatter: logging.Formatter, log_level: int, verbose: bool
    ) -> Handler:
        """Create a console handler."""
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
        return console_handler

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
