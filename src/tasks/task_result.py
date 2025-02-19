"""Abstract class for result data."""

from typing import Any

from entities.log_manager import LogManager
from util.math_utils import round_nested


class TaskResult:
    """
    Base class for encapsulating result data.

    This class provides functionality to store, process, and validate result data.
    The data is maintained as a dictionary and can be accessed or modified using
    dictionary-like syntax. It also offers built-in processing (e.g., rounding of
    numerical values) and validation (e.g., removal of None values) routines.
    """

    def __init__(self) -> None:
        """Initialize the TaskResult with an empty result data dictionary."""
        self._result_data: dict[str, Any] = {}
        self.logger = LogManager.get_logger(__name__)

    def process_results(self) -> None:
        """
        Process the result data.

        Rounds all numerical values (even within nested structures) to 2 decimal places.
        """
        self.logger.info("Processing result data.")
        for key, value in self._result_data.items():
            self._result_data[key] = round_nested(value, 2)

    def validate_results(self) -> None:
        """
        Validate the result data.

        Removes entries from the result data dictionary that have a value of None.
        """
        self.logger.info("Validating result data.")
        self._result_data = {
            key: value for key, value in self._result_data.items() if value is not None
        }

    def get_data(self) -> dict[str, Any]:
        """
        Return a copy of the result data.

        Returns:
            dict[str, Any]: A shallow copy of the result data dictionary.
        """
        return self._result_data.copy()

    def _get(self, key: str) -> Any:
        """
        Retrieve a value from the result data by key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key, or None if not found.
        """
        return self._result_data.get(key, None)

    def _set(self, key: str, value: Any) -> None:
        """
        Set a value in the result data for a given key.

        Args:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        self._result_data[key] = value

    def __str__(self) -> str:
        """Return the string representation of the result data."""
        return str(self._result_data)

    def __repr__(self) -> str:
        """Return the official string representation of the result data."""
        return str(self._result_data)

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve a value using dictionary-like access.

        Args:
            key (str): The key to retrieve from the result data.

        Returns:
            Any: The value associated with the key.
        """
        return self._get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a value using dictionary-like assignment.

        Args:
            key (str): The key to set in the result data.
            value (Any): The value to associate with the key.
        """
        self._set(key, value)
