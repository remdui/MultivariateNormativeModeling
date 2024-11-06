"""Abstract class for result data."""

from abc import ABC, abstractmethod
from typing import Any


class AbstractResult(ABC):
    """Abstract class for result data."""

    def __init__(self) -> None:
        """Initialize the AbstractResult."""
        self._result_data: dict[str, Any] = {}

    @abstractmethod
    def process_results(self) -> None:
        """Abstract method to process the result data."""

    @abstractmethod
    def validate_results(self) -> None:
        """Abstract method to validate the result data."""

    def get(self, key: str) -> Any:
        """Get a key from the result data.

        Args:
            key (str): The key to retrieve from the result data
        Returns:
            Any: The value associated with the key
        """
        return self._result_data.get(key, None)

    def set(self, key: str, value: Any) -> None:
        """Set a key in the result data.

        Args:
            key (str): The key to set in the result data
            value (Any): The value to associate with the key
        """
        self._result_data[key] = value

    def __str__(self) -> str:
        """Return the string representation of the result data."""
        return str(self._result_data)

    def __repr__(self) -> str:
        """Return the string representation of the result data."""
        return str(self._result_data)

    def __getitem__(self, key: str) -> Any:
        """Get a key from the result.

        Args:
            key (str): The key to retrieve from the result data
        Returns:
            Any: The value associated with the key
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a key in the result data.

        Args:
            key (str): The key to set in the result data
            value (Any): The value to associate with the key
        """
        self.set(key, value)
