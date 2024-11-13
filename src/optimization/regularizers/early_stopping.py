"""Early stopping to terminate training when validation loss does not improve."""

from entities.properties import Properties


class EarlyStopping:
    """Early stopping to terminate training when validation loss does not improve."""

    def __init__(self) -> None:
        """Initialize early stopping."""
        self.properties = Properties.get_instance()
        self.patience = self.properties.train.early_stopping.patience
        self.min_delta = self.properties.train.early_stopping.min_delta

        self.best_val_loss = float("inf")
        self.no_improvement_epochs = 0

    def stop_condition_met(self, val_loss: float) -> bool:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        return self.no_improvement_epochs >= self.patience

    def reset(self) -> None:
        """Reset the early stopping criteria."""
        self.best_val_loss = float("inf")
        self.no_improvement_epochs = 0
