"""Metrics for latent space analysis."""

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


def median_bandwidth(x: Any, sample_size: Any = None) -> float:
    """
    Compute the median heuristic bandwidth for matrix x.

    x: 2D numpy array (n x d)
    """
    n = x.shape[0]
    if sample_size is not None and sample_size < n:
        indices = np.random.choice(n, size=sample_size, replace=False)
        x_sample = x[indices]
    else:
        x_sample = x
    dists = pdist(x_sample, metric="euclidean")
    med = np.median(dists)
    return med if med > 0 else 0.001


def gaussian_grammat(x: Any, bw: Any) -> Any:
    """
    Compute the Gaussian kernel (Gram) matrix for x using bandwidth bw.

    x: 2D numpy array (n x d)
    """
    dists = squareform(pdist(x, metric="sqeuclidean"))
    K = np.exp(-dists / (2 * bw**2))
    return K


def discrete_grammat(x: Any) -> Any:
    """
    Compute the discrete kernel Gram matrix for x.

    For each pair (i,j), returns 1 if the observations are identical, else 0.
    """
    n = x.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = 1 if np.array_equal(x[i], x[j]) else 0
            K[i, j] = val
            K[j, i] = val
    return K


def custom_grammat(x: Any, fun: Any) -> Any:
    """
    Compute a Gram matrix for x using a custom kernel function.

    fun should take two 1D arrays (observations) and return a scalar.
    """
    n = x.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = fun(x[i], x[j])
            K[i, j] = val
            K[j, i] = val
    return K


def dhsic(
    X: Any,
    Y: Any,
    kernel: Any = "gaussian",
    bandwidth: Any = 1,
    matrix_input: Any = False,
) -> Any:
    """Compute the d-variable HSIC estimator."""
    time.time()

    # If matrix_input is True, split the matrix into its columns.
    if matrix_input:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array when matrix_input=True.")
        X = [X[:, i].reshape(-1, 1) for i in range(X.shape[1])]
    elif not isinstance(X, list):
        # If Y is provided, we treat X and Y as two separate variables.
        X = [X, Y]

    # Ensure each variable is a 2D array.
    d = len(X)
    for j in range(d):
        X[j] = np.atleast_2d(X[j])
        if X[j].shape[0] < X[j].shape[1]:
            X[j] = X[j].T
    n = X[0].shape[0]

    if n < 2 * d:
        return {"dHSIC": 0, "time": (0, 0), "bandwidth": None}

    # Ensure kernel and bandwidth are lists.
    if isinstance(kernel, str):
        kernel = [kernel] * d
    if isinstance(bandwidth, (int, float)):
        bandwidth = [bandwidth] * d

    # Compute Gram matrices.
    K_list = [None] * d
    start_gram = time.time()
    for j in range(d):
        if kernel[j] == "gaussian":
            bw = median_bandwidth(X[j])
            bandwidth[j] = bw
            K_list[j] = gaussian_grammat(X[j], bw)
        elif kernel[j] == "gaussian.fixed":
            K_list[j] = gaussian_grammat(X[j], bandwidth[j])
        elif kernel[j] == "discrete":
            bandwidth[j] = None
            K_list[j] = discrete_grammat(X[j])
        elif callable(kernel[j]):
            K_list[j] = custom_grammat(X[j], kernel[j])
        else:
            raise ValueError(
                "kernel must be 'gaussian', 'gaussian.fixed', 'discrete', or a callable"
            )

    timeGramMat = time.time() - start_gram

    # Compute dHSIC statistic.
    start_hsic = time.time()
    term1 = np.ones((n, n))
    term2 = 1.0
    term3 = np.full((n,), 2.0 / n)

    for j in range(d):
        term1 *= K_list[j]
        term2 *= np.sum(K_list[j])
        term2 = term2 / (n**2)
        term3 = (term3 / n) * np.sum(K_list[j], axis=0)
    dHSIC_val = (np.sum(term1) / (n**2)) + term2 - np.sum(term3)
    timeHSIC = time.time() - start_hsic

    return {"dHSIC": dHSIC_val, "time": (timeGramMat, timeHSIC), "bandwidth": bandwidth}


def calculate_latent_regression_error(engine: Any, covariate: str) -> dict[str, float]:
    """Perform latent regression on the latent space and the covariate."""
    logger = engine.logger
    if covariate in engine.recon_df.columns:
        df = engine.recon_df
    elif covariate in engine.test_df.columns:
        unique_id = engine.properties.dataset.unique_identifier_column
        if unique_id in engine.recon_df.columns and unique_id in engine.test_df.columns:
            df = pd.merge(
                engine.recon_df, engine.test_df[[unique_id, covariate]], on=unique_id
            )
        else:
            logger.error(
                "Unique identifier column not found in both recon_df and test_df for covariate regression."
            )
            return {}
    else:
        logger.error(f"Covariate '{covariate}' not found in recon_df or test_df.")
        return {}

    if not pd.api.types.is_numeric_dtype(df[covariate]):
        logger.error(
            f"Covariate '{covariate}' is not numeric. Regression cannot be performed."
        )
        return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error(
            "No latent space columns found with prefix 'z_mean_' for regression."
        )
        return {}

    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    logger.info(
        f"Latent covariate regression for '{covariate}': MSE={mse_val:.4f}, R²={r2_val:.4f}"
    )
    return {"mse": mse_val, "r2": r2_val}


def calculate_latent_dhsic(engine: Any, covariate: str) -> dict[str, float]:
    """
    Test nonlinear dependence between the latent representation and a single covariate.

    using an alternative dHSIC implementation.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate to test (must be in test_df).

    Returns:
        dict[str, float]: Dictionary with dHSIC statistic and timing information.
                          (Note: This implementation does not compute a permutation-based p-value.)
    """
    logger = engine.logger

    if covariate not in engine.test_df.columns:
        logger.error(f"Covariate '{covariate}' not found in test_df.")
        return {}

    # Extract latent representation (assume columns with prefix "z_mean_")
    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    # Use the reconstruction data for latent representation
    Z = engine.recon_df[latent_cols].to_numpy()

    # Extract covariate values
    y = engine.test_df[covariate]
    # For categorical data, you might want to encode, but here we assume numeric
    y = y.to_numpy().reshape(-1, 1)

    # Check matching shape
    if Z.shape[0] != y.shape[0]:
        logger.error(
            f"Shape mismatch: latent ({Z.shape[0]}) vs covariate '{covariate}' ({y.shape[0]})."
        )
        return {}

    # Compute dHSIC using the alternative implementation.
    # Here we use a Gaussian kernel with default settings.
    result = dhsic(Z, y, kernel="gaussian", bandwidth=1, matrix_input=False)
    stat = result["dHSIC"]
    gram_time, hsic_time = result["time"]
    used_bandwidth = result["bandwidth"]

    logger.info(
        f"Alternative dHSIC for covariate '{covariate}': "
        f"statistic={stat:.6f}, Gram time={gram_time:.4f}s, HSIC time={hsic_time:.4f}s, "
        f"bandwidth={used_bandwidth}"
    )

    return {"dhsic_statistic": stat, "gram_time": gram_time, "hsic_time": hsic_time}
