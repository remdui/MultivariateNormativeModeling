"""Mean Squared Error (MSE) computation for tensors."""

from torch import Tensor


def compute_mse(
    original: Tensor,
    reconstructed: Tensor,
    mse_type: str = "total",
) -> Tensor:
    """Compute Mean Squared Error (MSE) for different types.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        mse_type (str): Type of MSE to compute. Options are 'sample', 'feature', 'total', 'covariate'.

    Returns:
        Tensor: Computed MSE based on the specified type.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate squared error
    squared_error = (original - reconstructed) ** 2

    # MSE per sample
    if mse_type == "sample":
        # Average over features for each sample
        mse = squared_error.mean(dim=1)

    # MSE per feature
    elif mse_type == "feature":
        # Average over samples for each feature
        mse = squared_error.mean(dim=0)

    # MSE total
    elif mse_type == "total":
        # Average over all samples and features
        mse = squared_error.mean()

    else:
        raise ValueError(
            f"Invalid mse_type: {mse_type}. Choose from 'sample', 'feature', 'total'."
        )

    return mse
