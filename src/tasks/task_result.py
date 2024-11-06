"""Abstract class for result data."""

from typing import Any

from entities.log_manager import LogManager


class TaskResult:
    """Abstract class for result data."""

    def __init__(self) -> None:
        """Initialize the AbstractResult."""
        self.__result_data: dict[str, Any] = {}
        self.logger = LogManager.get_logger(__name__)

    def process_results(self) -> None:
        """Process the result data."""
        self.logger.info("Processing the validation results.")

        # Round all the values to 2 decimal places
        for key, value in self.__result_data.items():
            self.__result_data[key] = round(value, 2)

    def validate_results(self) -> None:
        """Validate the result data."""
        self.logger.info("Validating the validation results.")

        # Remove keys with None values
        self.__result_data = {
            key: value for key, value in self.__result_data.items() if value is not None
        }

    def get_data(self) -> dict[str, Any]:
        """Return a copy of the result data dictionary."""
        return self.__result_data.copy()

    def __get(self, key: str) -> Any:
        """Get a key from the result data.

        Args:
            key (str): The key to retrieve from the result data
        Returns:
            Any: The value associated with the key
        """
        return self.__result_data.get(key, None)

    def __set(self, key: str, value: Any) -> None:
        """Set a key in the result data.

        Args:
            key (str): The key to set in the result data
            value (Any): The value to associate with the key
        """
        self.__result_data[key] = value

    def __str__(self) -> str:
        """Return the string representation of the result data."""
        return str(self.__result_data)

    def __repr__(self) -> str:
        """Return the string representation of the result data."""
        return str(self.__result_data)

    def __getitem__(self, key: str) -> Any:
        """Get a key from the result.

        Args:
            key (str): The key to retrieve from the result data
        Returns:
            Any: The value associated with the key
        """
        return self.__get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a key in the result data.

        Args:
            key (str): The key to set in the result data
            value (Any): The value to associate with the key
        """
        self.__set(key, value)
