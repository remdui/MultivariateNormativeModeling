"""NRMSE metric for comparing original and reconstructed data."""

from torch import Tensor

from analysis.tools.metrics.rmse import compute_rmse


def compute_nrmse(
    original: Tensor,
    reconstructed: Tensor,
    metric_type: str = "feature",
    normalization_type: str = "range",
) -> Tensor:
    """Compute Normalized Root Mean Square Error (NRMSE) for tensors.

    NRMSE normalizes RMSE by the range or mean of the original data, providing a scale-independent measure.

    Equation: NRMSE = RMSE / (max(original) - min(original)) or RMSE / mean(original)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        metric_type (str): Type of NRMSE to compute. Options are 'sample', 'feature', 'total'.
        normalization_type (str): Normalization type, either 'range' or 'mean'.

    Returns:
        Tensor: Computed NRMSE.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate the RMSE
    rmse = compute_rmse(original, reconstructed, metric_type=metric_type)

    # Normalize by the range of the original data
    if normalization_type == "range":
        data_range = original.max() - original.min()
        if data_range == 0:
            raise ValueError(
                "Range of original data is zero; cannot compute NRMSE with 'range' normalization."
            )
        nrmse = rmse / data_range

    # Normalize by the mean of the original data
    elif normalization_type == "mean":
        data_mean = original.mean()
        if data_mean == 0:
            raise ValueError(
                "Mean of original data is zero; cannot compute NRMSE with 'mean' normalization."
            )
        nrmse = rmse / data_mean

    else:
        raise ValueError(
            f"Invalid normalization_type: {normalization_type}. Choose from 'range' or 'mean'."
        )

    return nrmse
