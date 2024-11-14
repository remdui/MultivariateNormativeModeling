"""Mean Bias Deviation (MBD) metric for detecting systematic bias in reconstructed data."""

from torch import Tensor


def compute_mbd(original: Tensor, reconstructed: Tensor) -> Tensor:
    """Compute Mean Bias Deviation (MBD) per feature.

    MBD detects systematic bias by indicating whether reconstructed values consistently overestimate or underestimate the original values for each feature

    Equation: MBD = Σ (reconstructed - original) / num_samples

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: Mean Bias Deviation for each feature, indicating systematic bias.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate the difference between reconstructed and original values
    bias = reconstructed - original

    # Mean Bias Deviation per feature
    mbd = bias.mean(dim=0)

    return mbd
