"""Metrics for latent space analysis."""

from typing import Any

import numpy as np
import pandas as pd
from hyppo.d_variate import dHsic  # type: ignore
from scipy import stats  # type: ignore
from sklearn.cross_decomposition import CCA  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.feature_selection import mutual_info_regression  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor  # type: ignore


def calculate_latent_dhsic(engine: Any, covariate: str) -> dict[str, float]:
    """
    Measure the dependence between latent space and a single covariate using dHSIC.

    Args:
        engine (Any): The analysis engine with loaded latent and covariate data.
        covariate (str): Covariate column name to test against latent space.

    Returns:
        dict[str, float]: Dictionary with dHSIC statistic and p-value.
    """
    logger = engine.logger

    if covariate not in engine.test_df.columns:
        logger.error(f"Covariate '{covariate}' not found in test_df.")
        return {}

    # Get latent variables
    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    latent_data = engine.recon_df[latent_cols].to_numpy()

    # Get covariate values
    col_data = engine.test_df[covariate]
    cov_values = col_data.to_numpy().reshape(-1, 1)

    dhsic_test = dHsic()
    stat, p_value = dhsic_test.test(latent_data, cov_values, workers=1)

    logger.info(
        f"dHSIC test for covariate '{covariate}': statistic={stat:.6f}, p-value={p_value:.6f}"
    )
    return {"dhsic_statistic": stat, "p_value": p_value}


def calculate_latent_mutual_information(engine: Any, covariate: str) -> dict[str, Any]:
    """
    Calculate the mutual information between the latent representation and a given covariate.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate to analyze (must be in test_df).

    Returns:
        dict[str, float]: Dictionary containing the mutual information for each latent dimension
                          and the total mutual information (summed across dimensions).
    """
    logger = engine.logger

    if covariate not in engine.test_df.columns:
        logger.error(f"Covariate '{covariate}' not found in test_df.")
        return {}

    # Identify latent representation columns (expected to have prefix "z_mean_")
    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    # Extract latent representation and covariate data
    Z = engine.recon_df[latent_cols].to_numpy()
    y = engine.test_df[covariate]

    y = y.to_numpy()

    # Compute mutual information for each latent dimension
    mi = mutual_info_regression(Z, y, random_state=42)

    total_mi = float(mi.sum())
    logger.info(
        f"Mutual information for covariate '{covariate}': total MI = {total_mi:.6f}"
    )
    return {"mutual_info_per_dim": list(mi), "total_mutual_info": total_mi}


def calculate_latent_regression_error(engine: Any, covariate: str) -> dict[str, float]:
    """
    Perform a linear regression of the covariate on the latent space (z_mean_* columns).

    Returns MSE, MAE, R², and explained variance.

    Args:
        engine (Any): Analysis engine or any object containing recon_df/test_df dataframes.
        covariate (str): Name of the covariate column.

    Returns:
        dict[str, float]: Dictionary with keys: 'mse', 'mae', 'r2', 'explained_variance'.
    """
    logger = engine.logger

    # Identify which dataframe contains the covariate; if needed, merge using a unique ID.
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
                "Unique identifier column not found in both recon_df and test_df "
                f"for covariate '{covariate}' regression."
            )
            return {}
    else:
        logger.error(f"Covariate '{covariate}' not found in recon_df or test_df.")
        return {}

    # Check if covariate is numeric
    if not pd.api.types.is_numeric_dtype(df[covariate]):
        logger.error(
            f"Covariate '{covariate}' is not numeric. Regression cannot be performed."
        )
        return {}

    # Extract latent columns
    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    # Prepare data
    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a simple linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Compute metrics
    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    # Log and return results
    logger.info(
        f"Latent covariate regression for '{covariate}': "
        f"MSE={mse_val:.4f}, MAE={mae_val:.4f}, R²={r2_val:.4f}, "
        f"ExplainedVar={explained_var:.4f}"
    )

    return {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "explained_variance": explained_var,
    }


def calculate_latent_correlation_coefficients(
    engine: Any, covariate: str
) -> dict[str, list[float]]:
    """
    Compute Pearson and Spearman correlation coefficients between each latent dimension and a given covariate.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate (must be in test_df).

    Returns:
        dict[str, list[float]]: Dictionary with keys 'pearson' and 'spearman' mapping to lists of correlation coefficients for each latent dimension.
    """
    logger = engine.logger

    if covariate not in engine.test_df.columns:
        logger.error(f"Covariate '{covariate}' not found in test_df.")
        return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    # Extract latent representation from recon_df and covariate values from test_df
    X = engine.recon_df[latent_cols]
    y = engine.test_df[covariate]
    pearson_coeffs = []
    spearman_coeffs = []

    for col in X.columns:
        # Compute Pearson correlation
        pearson_corr, _ = stats.pearsonr(X[col], y)
        pearson_coeffs.append(pearson_corr)

        # Compute Spearman correlation
        spearman_corr, _ = stats.spearmanr(X[col], y)
        spearman_coeffs.append(spearman_corr)

    logger.info(f"Pearson correlations: {pearson_coeffs}")
    logger.info(f"Spearman correlations: {spearman_coeffs}")
    return {"pearson": pearson_coeffs, "spearman": spearman_coeffs}


def calculate_latent_nonlinear_regression_error(
    engine: Any, covariate: str
) -> dict[str, float]:
    """
    Perform nonlinear regression using a RandomForestRegressor to predict the covariate from the latent representation.

    Evaluate the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), R², and explained variance score.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate to predict.

    Returns:
        dict[str, float]: Dictionary with 'mae', 'mse', 'r2', and 'explained_variance' scores.
    """
    logger = engine.logger

    # Determine data source: if covariate exists in recon_df, use it directly; otherwise, merge using a unique identifier.
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
                "Unique identifier column not found in both recon_df and test_df for regression."
            )
            return {}
    else:
        logger.error(f"Covariate '{covariate}' not found in recon_df or test_df.")
        return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Nonlinear regression using RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    logger.info(
        f"Nonlinear regression for '{covariate}': "
        f"MAE={mae:.4f}, MSE={mse_val:.4f}, R²={r2_val:.4f}, Explained Var={explained_var:.4f}"
    )
    return {
        "mae": mae,
        "mse": mse_val,
        "r2": r2_val,
        "explained_variance": explained_var,
    }


def calculate_latent_cca_single(engine: Any, covariate: str) -> dict[str, float]:
    """
    Perform Canonical Correlation Analysis (CCA) between the latent representation and a single covariate.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate from test_df to analyze.

    Returns:
        dict[str, float]: Dictionary with the canonical correlation coefficient.
    """
    logger = engine.logger

    # Verify that the covariate exists in test_df.
    if covariate not in engine.test_df.columns:
        logger.error(f"Covariate '{covariate}' not found in test_df.")
        return {}

    # Extract latent representation from recon_df (using columns starting with "z_mean_")
    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    # Get the latent data and reshape the covariate to a 2D array.
    X = engine.recon_df[latent_cols].to_numpy()
    Y = engine.test_df[covariate].to_numpy().reshape(-1, 1)

    # Set up CCA with one component since we're using one covariate.
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(X, Y)

    # Compute the canonical correlation between the first (and only) components.
    corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    logger.info(f"Canonical correlation for covariate '{covariate}': {corr:.6f}")
    return {"canonical_correlation": corr}


def calculate_latent_adversarial_performance(
    engine: Any,
    covariate: str,
    task_type: str = "regression",  # or "classification"
    hidden_layer_sizes: tuple = (32, 32),
    max_iter: int = 10000,
) -> dict[str, float]:
    """
    Train an adversarial predictor (MLP) to predict the given covariate from.

    the latent representation z_mean_*. This is useful as a measure of how
    'covariate-invariant' the latent space truly is. If the covariate can be
    accurately predicted, it indicates that the latent space is not invariant.

    Args:
        engine (Any): The analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate to predict. Can be numeric or categorical.
        task_type (str): "regression" or "classification". Defaults to "regression".
        hidden_layer_sizes (tuple): Sizes of hidden layers in the MLP.
        max_iter (int): Maximum number of iterations (epochs) for training the MLP.

    Returns:
        dict[str, float]: If regression, returns MSE, MAE, R², and explained variance.
                          If classification, returns accuracy and F1 score.
    """
    logger = engine.logger

    # Identify where the covariate resides (recon_df or test_df), then merge if needed
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
                f"Unique identifier column '{unique_id}' not found in both recon_df and test_df."
            )
            return {}
    else:
        logger.error(f"Covariate '{covariate}' not found in recon_df or test_df.")
        return {}

    # Extract latent space columns
    latent_cols = [col for col in df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task_type == "regression":
        # Ensure covariate is numeric
        if not pd.api.types.is_numeric_dtype(df[covariate]):
            logger.error(
                f"Covariate '{covariate}' is not numeric but task_type='regression'."
            )
            return {}

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=0.0001,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        mse_val = mean_squared_error(y_test, y_pred)
        mae_val = mean_absolute_error(y_test, y_pred)
        r2_val = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)

        logger.info(
            f"Adversarial MLP (regression) for '{covariate}': "
            f"MSE={mse_val:.4f}, MAE={mae_val:.4f}, R²={r2_val:.4f}, "
            f"ExplainedVar={explained_var:.4f}"
        )
        return {
            "mse": mse_val,
            "mae": mae_val,
            "r2": r2_val,
            "explained_variance": explained_var,
        }

    if task_type == "classification":
        # For classification, convert covariate to integer categories if needed
        if pd.api.types.is_numeric_dtype(df[covariate]):
            # If numeric but you truly want classification, you need a binarization or discretization step
            logger.error(
                f"Covariate '{covariate}' is numeric, but task_type='classification'. "
                "Please discretize or encode the covariate before classification."
            )
            return {}

        # Convert to categorical codes
        y_train_cat = pd.Categorical(y_train).codes
        y_test_cat = pd.Categorical(y_test).codes

        model_class = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
        )
        model_class.fit(X_train, y_train_cat)
        y_pred_cat = model_class.predict(X_test)

        # Compute classification metrics
        acc_val = accuracy_score(y_test_cat, y_pred_cat)
        f1_val = f1_score(y_test_cat, y_pred_cat, average="weighted")

        logger.info(
            f"Adversarial MLP (classification) for '{covariate}': "
            f"Accuracy={acc_val:.4f}, F1={f1_val:.4f}"
        )
        return {"accuracy": acc_val, "f1": f1_val}

    logger.error(
        f"Unknown task_type '{task_type}'. Use 'regression' or 'classification'."
    )
    return {}
