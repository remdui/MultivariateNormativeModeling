"""Structural Similarity Index (SSIM) for tabular data."""

from torch import Tensor


def compute_ssim(
    original: Tensor, reconstructed: Tensor, c1: float = 1e-4, c2: float = 9e-4
) -> Tensor:
    """
    Compute Structural Similarity Index (SSIM) per feature for tabular data.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        c1 (float): Small constant to stabilize luminance calculation.
        c2 (float): Small constant to stabilize contrast calculation.

    Returns:
        Tensor: SSIM for each feature.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Compute means
    mu_x = original.mean(dim=0)
    mu_y = reconstructed.mean(dim=0)

    # Compute variances and covariance
    sigma_x = original.var(dim=0, unbiased=False)
    sigma_y = reconstructed.var(dim=0, unbiased=False)
    sigma_xy = ((original - mu_x) * (reconstructed - mu_y)).mean(dim=0)

    # Calculate SSIM per feature
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )

    return ssim
