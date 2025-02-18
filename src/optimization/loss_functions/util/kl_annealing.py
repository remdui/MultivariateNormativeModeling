"""KL Annealing Class."""


class KLAnnealing:
    """Helper class to handle KL-weight annealing logic."""

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        kl_anneal_start: int,
        kl_anneal_end: int,
    ) -> None:
        """
        Args:

            beta_start (float): initial KL weight.
            beta_end (float): final KL weight.
            kl_anneal_start (int): epoch to begin annealing.
            kl_anneal_end (int): epoch to finish annealing.
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

    def compute_beta(self, current_epoch: int) -> float:
        """Linearly anneal beta from beta_start to beta_end.

        over [kl_anneal_start, kl_anneal_end].
        """
        # If the annealing range is invalid or not set, just return final beta
        if self.kl_anneal_end <= self.kl_anneal_start:
            return self.beta_end

        # Before the start of annealing
        if current_epoch < self.kl_anneal_start:
            return self.beta_start

        # After the end of annealing
        if current_epoch >= self.kl_anneal_end:
            return self.beta_end

        # In the middle of the annealing range
        progress = (current_epoch - self.kl_anneal_start) / float(
            self.kl_anneal_end - self.kl_anneal_start
        )
        return self.beta_start + progress * (self.beta_end - self.beta_start)
