"""Utility functions and classes for logging.

This module configures logging based on application properties and provides a centralized
LogManager for obtaining logger instances. It supports both file and console logging with
rotating file handlers.
"""

import logging
import os
from logging import Handler
from logging.handlers import RotatingFileHandler

from entities.properties import Properties
from util.errors import ConfigurationError


class LogManager:
    """Manages logging configuration based on application properties."""

    _logger_initialized = False

    @staticmethod
    def setup_logging() -> None:
        """
        Set up logging configuration using settings from the Properties singleton.

        Retrieves log directory, debug/verbose flags, and log level from Properties.
        Configures both file and console handlers. If logging is already configured,
        further attempts are skipped.
        """
        if LogManager._logger_initialized:
            return  # Prevent reconfiguration if already set up

        try:
            properties = Properties.get_instance()
            log_dir = properties.system.log_dir
            debug = properties.general.debug
            verbose = properties.general.verbose
            log_level_str = properties.general.log_level.upper()

            # Define the log file path.
            log_file = os.path.join(log_dir, "application.log")

            # Determine the log level; verbose mode captures all messages.
            log_level = (
                logging.NOTSET
                if verbose
                else getattr(logging, log_level_str, logging.INFO)
            )

            # Create a common formatter for all handlers.
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
            )

            # Create and configure handlers.
            file_handler = LogManager.__create_file_handler(
                formatter, log_file, log_level, verbose
            )
            console_handler = LogManager.__create_console_handler(
                debug, formatter, log_level, verbose
            )

            # Configure the root logger: clear existing handlers and add new ones.
            root_logger = logging.getLogger()
            root_logger.handlers = []
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.NOTSET)  # Allow all messages to be processed.

            LogManager._logger_initialized = True

            # Log the configuration details.
            logger = LogManager.get_logger(__name__)
            logger.info(
                f"Logging initialized: verbose={verbose}, debug={debug}, log_level={log_level_str}"
            )

        except ConfigurationError:
            # If Properties are not yet initialized or any configuration error occurs,
            # fallback to a basic logging configuration.
            logging.getLogger(__name__).warning(
                "Properties have not been initialized. Using basic logging configuration temporarily."
            )
            LogManager._logger_initialized = (
                True  # Prevent further reconfiguration attempts.
            )

    @staticmethod
    def __create_file_handler(
        formatter: logging.Formatter, log_file: str, log_level: int, verbose: bool
    ) -> Handler:
        """
        Create and configure a rotating file handler.

        Args:
            formatter (logging.Formatter): Formatter to format log messages.
            log_file (str): Path to the log file.
            log_level (int): Log level for the handler.
            verbose (bool): If True, the handler captures all log messages.

        Returns:
            Handler: Configured rotating file handler.
        """
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=100
        )
        file_handler.setLevel(logging.NOTSET if verbose else log_level)
        file_handler.setFormatter(formatter)

        # Immediately rotate the log file if it already exists.
        if os.path.exists(log_file):
            file_handler.doRollover()

        return file_handler

    @staticmethod
    def __create_console_handler(
        debug: bool, formatter: logging.Formatter, log_level: int, verbose: bool
    ) -> Handler:
        """
        Create and configure a console (stream) handler.

        Args:
            debug (bool): Debug flag to determine handler level.
            formatter (logging.Formatter): Formatter to format log messages.
            log_level (int): Log level for the handler.
            verbose (bool): If True, the handler captures all log messages.

        Returns:
            Handler: Configured console handler.
        """
        console_handler = logging.StreamHandler()
        if verbose:
            console_handler.setLevel(logging.NOTSET)
        elif debug:
            console_handler.setLevel(log_level)
        else:
            # Effectively disable console output by setting a level above CRITICAL.
            console_handler.setLevel(logging.CRITICAL + 1)
        console_handler.setFormatter(formatter)
        return console_handler

    @staticmethod
    def reconfigure_logging() -> None:
        """
        Reconfigure logging based on updated Properties.

        Resets the initialization flag and sets up logging again.
        """
        LogManager._logger_initialized = False
        LogManager.setup_logging()

    @staticmethod
    def get_logger(name: str = "") -> logging.Logger:
        """
        Get a configured logger instance with the specified name.

        Args:
            name (str): Name of the logger. Defaults to the root logger if empty.

        Returns:
            logging.Logger: Logger instance configured according to application settings.
        """
        LogManager.setup_logging()
        return logging.getLogger(name)
