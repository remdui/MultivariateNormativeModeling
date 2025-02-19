"""Early stopping utility to terminate training when validation loss stagnates.

This module provides the EarlyStopping class, which monitors the validation loss during training.
Training is halted if the loss does not improve by at least a minimum threshold (min_delta)
for a specified number of consecutive epochs (patience).
"""

from entities.properties import Properties


class EarlyStopping:
    """
    Early stopping mechanism to halt training when validation loss shows no significant improvement.

    The criterion is based on comparing the current validation loss to the best observed loss.
    If the improvement is less than 'min_delta' for 'patience' consecutive epochs, the stop
    condition is met.
    """

    def __init__(self) -> None:
        """
        Initialize the EarlyStopping instance.

        Retrieves early stopping parameters from the global Properties instance:
            - patience: Maximum number of consecutive epochs without significant improvement.
            - min_delta: Minimum required improvement in validation loss to reset the counter.
        """
        self.properties = Properties.get_instance()
        self.patience = self.properties.train.early_stopping.patience
        self.min_delta = self.properties.train.early_stopping.min_delta

        self.best_val_loss = float("inf")
        self.no_improvement_epochs = 0

    def stop_condition_met(self, val_loss: float) -> bool:
        """
        Check if training should be stopped based on the current validation loss.

        Updates internal state by resetting the no-improvement counter if a significant improvement
        is observed; otherwise, increments the counter. Training should be stopped when the counter
        reaches the 'patience' threshold.

        Args:
            val_loss (float): The current validation loss.

        Returns:
            bool: True if training should be halted, False otherwise.
        """
        # Check if the current loss is sufficiently lower than the best recorded loss.
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        return self.no_improvement_epochs >= self.patience

    def reset(self) -> None:
        """
        Reset the early stopping state.

        Resets the best validation loss and the counter of consecutive epochs with no improvement.
        Use this to restart early stopping for a new training run.
        """
        self.best_val_loss = float("inf")
        self.no_improvement_epochs = 0
