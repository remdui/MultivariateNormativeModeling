"""Defines the AbstractDataConverter abstract base class."""

from abc import ABC, abstractmethod


class AbstractDataConverter(ABC):
    """Abstract base class for data converters."""

    @abstractmethod
    def convert(self, input_file_name: str, output_file_name: str) -> None:
        """Convert data from one format to another.

        Args:
            input_file_name (str): File name of the input raw data.
            output_file_name (str): File name of the output processed data.
        """
