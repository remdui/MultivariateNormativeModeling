"""Enumeration for the different phases of data exploration."""

import enum


class DataExplorationPhase(enum.Enum):
    """Enumeration for the different phases of data exploration."""

    RAW_DATA = 0
    PROCESSED_DATA = 1
    OUTPUT_DATA = 2

    def __str__(self) -> str:
        """Get the string representation of the enumeration."""
        return self.name.lower()

    def __repr__(self) -> str:
        """Get the string representation of the enumeration."""
        return self.__str__()
