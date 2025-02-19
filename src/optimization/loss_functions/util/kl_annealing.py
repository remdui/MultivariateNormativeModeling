"""KL Annealing Class.

This module defines the KLAnnealing class, which computes a KL divergence weight (beta)
that is linearly annealed from an initial value (beta_start) to a final value (beta_end)
over a specified range of epochs. This weight is typically used in variational autoencoders
to gradually increase the influence of the KL divergence term during training.
"""


class KLAnnealing:
    """
    Helper class to compute a linearly annealed KL divergence weight (beta).

    Beta is annealed from beta_start to beta_end between kl_anneal_start and kl_anneal_end epochs.
    - Returns beta_start if the current epoch is before kl_anneal_start.
    - Returns beta_end if the current epoch is at or after kl_anneal_end, or if the annealing range is invalid.
    - Otherwise, returns an interpolated beta value.
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        kl_anneal_start: int,
        kl_anneal_end: int,
    ) -> None:
        """
        Initialize the KLAnnealing instance.

        Args:
            beta_start (float): The initial KL divergence weight.
            beta_end (float): The final KL divergence weight.
            kl_anneal_start (int): The epoch at which annealing begins.
            kl_anneal_end (int): The epoch at which annealing ends.
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

    def compute_beta(self, current_epoch: int) -> float:
        """
        Compute the current KL divergence weight (beta) based on the current epoch.

        Returns:
            float: The annealed KL divergence weight.
                   - beta_start if current_epoch is before kl_anneal_start.
                   - beta_end if current_epoch is at or after kl_anneal_end, or if the annealing range is invalid.
                   - A linearly interpolated value between beta_start and beta_end otherwise.

        Args:
            current_epoch (int): The current training epoch.
        """
        # If the annealing range is invalid, return beta_end.
        if self.kl_anneal_end <= self.kl_anneal_start:
            return self.beta_end

        # Before annealing starts, return beta_start.
        if current_epoch < self.kl_anneal_start:
            return self.beta_start

        # After annealing ends, return beta_end.
        if current_epoch >= self.kl_anneal_end:
            return self.beta_end

        # Compute the linear interpolation for epochs within the annealing range.
        progress = (current_epoch - self.kl_anneal_start) / (
            self.kl_anneal_end - self.kl_anneal_start
        )
        return self.beta_start + progress * (self.beta_end - self.beta_start)
