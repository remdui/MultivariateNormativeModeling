"""Metrics for latent space analysis."""

from typing import Any

import numpy as np
import pandas as pd
from hyppo.d_variate import dHsic  # type: ignore
from scipy import stats  # type: ignore
from sklearn.cross_decomposition import CCA  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (  # type: ignore
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor  # type: ignore

TEST_SIZE = 0.3


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

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    latent_data = engine.recon_df[latent_cols].to_numpy()
    cov_values = engine.test_df[covariate].to_numpy().reshape(-1, 1)

    dhsic_test = dHsic()
    stat, p_value = dhsic_test.test(latent_data, cov_values, workers=1)

    logger.info(
        f"dHSIC test for covariate '{covariate}': statistic={stat:.6f}, p-value={p_value:.6f}"
    )
    return {"dhsic_statistic": stat, "p_value": p_value}


def calculate_latent_mutual_information(engine: Any, covariate: str) -> dict[str, Any]:
    """
    Calculate the mutual information between the latent representation and a given covariate.

    For one-hot encoded covariates (e.g. sex, site), it will search for columns with prefix "covariate_" and
    convert them to categorical labels using np.argmax.

    Args:
        engine (Any): Analysis engine instance with loaded latent and test data.
        covariate (str): The name of the covariate to analyze.

    Returns:
        dict[str, Any]: Mutual information per latent dimension and the total mutual information.
    """
    logger = engine.logger

    # Get the target values.
    if covariate in engine.test_df.columns:
        y = engine.test_df[covariate].to_numpy()
    else:
        candidate_cols = [
            col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
        ]
        if candidate_cols:
            y = np.argmax(engine.test_df[candidate_cols].to_numpy(), axis=1)
        else:
            logger.error(
                f"Covariate '{covariate}' not found in test_df (neither direct nor one-hot encoded)."
            )
            return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    Z = engine.recon_df[latent_cols].to_numpy()

    # Decide whether to use regression or classification mutual information.
    if np.issubdtype(y.dtype, np.number) and (len(np.unique(y)) > 10):
        mi = mutual_info_regression(Z, y, random_state=42)
    else:
        mi = mutual_info_classif(Z, y, random_state=42)

    total_mi = float(mi.sum())
    logger.info(
        f"Mutual information for covariate '{covariate}': total MI = {total_mi:.6f}"
    )
    return {"mutual_info_per_dim": list(mi), "total_mutual_info": total_mi}


def calculate_latent_cca_single(engine: Any, covariate: str) -> dict[str, float]:
    """
    Perform Canonical Correlation Analysis (CCA) between the latent representation and a single covariate.

    For one-hot encoded covariates, candidate columns (e.g. "sex_", "site_") are converted via np.argmax.

    Args:
        engine (Any): Analysis engine with loaded latent and test data.
        covariate (str): The name of the covariate to analyze.

    Returns:
        dict[str, float]: Dictionary with the canonical correlation coefficient.
    """
    logger = engine.logger

    if covariate in engine.test_df.columns:
        Y = engine.test_df[covariate].to_numpy().reshape(-1, 1)
    else:
        candidate_cols = [
            col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
        ]
        if candidate_cols:
            Y = np.argmax(engine.test_df[candidate_cols].to_numpy(), axis=1).reshape(
                -1, 1
            )
        else:
            logger.error(
                f"Covariate '{covariate}' not found in test_df (neither direct nor one-hot encoded)."
            )
            return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = engine.recon_df[latent_cols].to_numpy()
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(X, Y)
    corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    logger.info(f"Canonical correlation for covariate '{covariate}': {corr:.6f}")
    return {"canonical_correlation": corr}


def calculate_latent_correlation_coefficients(
    engine: Any, covariate: str
) -> dict[str, list[float]]:
    """
    Compute Pearson and Spearman correlation coefficients between each latent dimension and a given covariate.

    For one-hot encoded covariates, candidate columns (e.g. "sex_", "site_") are combined by taking the argmax.

    Args:
        engine (Any): Analysis engine with loaded latent and test data.
        covariate (str): The name of the covariate.

    Returns:
        dict[str, list[float]]: Dictionary with lists of Pearson and Spearman correlation coefficients.
    """
    logger = engine.logger

    if covariate in engine.test_df.columns:
        y = engine.test_df[covariate]
    else:
        candidate_cols = [
            col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
        ]
        if candidate_cols:
            y = pd.Series(np.argmax(engine.test_df[candidate_cols].to_numpy(), axis=1))
        else:
            logger.error(
                f"Covariate '{covariate}' not found in test_df (neither direct nor one-hot encoded)."
            )
            return {}

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = engine.recon_df[latent_cols]
    pearson_coeffs = []
    spearman_coeffs = []

    for col in X.columns:
        pearson_corr, _ = stats.pearsonr(X[col], y)
        spearman_corr, _ = stats.spearmanr(X[col], y)
        pearson_coeffs.append(pearson_corr)
        spearman_coeffs.append(spearman_corr)

    logger.info(f"Pearson correlations: {pearson_coeffs}")
    logger.info(f"Spearman correlations: {spearman_coeffs}")
    return {"pearson": pearson_coeffs, "spearman": spearman_coeffs}


def calculate_latent_regression_error(engine: Any, covariate: str) -> dict[str, float]:
    """
    Perform a linear regression of the covariate on the latent space (z_mean_* columns).

    Returns MSE, MAE, R², and explained variance.

    Args:
        engine (Any): Analysis engine containing recon_df/test_df dataframes.
        covariate (str): Name of the covariate column.

    Returns:
        dict[str, float]: Dictionary with keys: 'mse', 'mae', 'r2', 'explained_variance'.
    """
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
                f"Unique identifier column not found in both recon_df and test_df for covariate '{covariate}'."
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
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = df[latent_cols].to_numpy()
    y = df[covariate].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    logger.info(
        f"Latent covariate regression for '{covariate}': MSE={mse_val:.4f}, MAE={mae_val:.4f}, "
        f"R²={r2_val:.4f}, ExplainedVar={explained_var:.4f}"
    )
    return {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "explained_variance": explained_var,
    }


def calculate_latent_nonlinear_regression_error(
    engine: Any, covariate: str
) -> dict[str, float]:
    """
    Perform nonlinear regression using a RandomForestRegressor to predict the covariate from the latent representation.

    Evaluate using MAE, MSE, R², and explained variance score.

    Args:
        engine (Any): Analysis engine with loaded latent and test data.
        covariate (str): The name of the covariate.

    Returns:
        dict[str, float]: Dictionary with 'mae', 'mse', 'r2', and 'explained_variance'.
    """
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
        X, y, test_size=TEST_SIZE, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    logger.info(
        f"Nonlinear regression for '{covariate}': MAE={mae:.4f}, MSE={mse_val:.4f}, "
        f"R²={r2_val:.4f}, ExplainedVar={explained_var:.4f}"
    )
    return {
        "mae": mae,
        "mse": mse_val,
        "r2": r2_val,
        "explained_variance": explained_var,
    }


def calculate_latent_adversarial_performance(
    engine: Any,
    covariate: str,
    task_type: str = "regression",  # or "classification"
    hidden_layer_sizes: tuple = (32, 32),
    max_iter: int = 10000,
) -> dict[str, float]:
    """
    Train an adversarial predictor (MLP) to predict the given covariate from the latent representation.

    Supports both regression and classification. For classification, if the covariate is one-hot encoded,
    the method converts it to categorical labels. The method then computes additional classification metrics such
    as AUC, precision, recall, F1 score, and the confusion matrix.

    Args:
        engine (Any): Analysis engine with loaded latent and test data.
        covariate (str): The covariate to predict.
        task_type (str): "regression" or "classification".
        hidden_layer_sizes (tuple): Hidden layer sizes for MLP.
        max_iter (int): Maximum iterations for the classifier.

    Returns:
        dict[str, float]: Metrics dictionary.
    """
    logger = engine.logger

    # Determine data source. First check if covariate exists directly; otherwise, look for one-hot encoded columns.
    if covariate in engine.recon_df.columns or covariate in engine.test_df.columns:
        if covariate in engine.recon_df.columns:
            df = engine.recon_df
        else:
            unique_id = engine.properties.dataset.unique_identifier_column
            if (
                unique_id in engine.recon_df.columns
                and unique_id in engine.test_df.columns
            ):
                df = pd.merge(
                    engine.recon_df,
                    engine.test_df[[unique_id, covariate]],
                    on=unique_id,
                )
            else:
                logger.error(
                    f"Unique identifier column not found for covariate '{covariate}'."
                )
                return {}
    else:
        candidate_cols = [
            col for col in engine.recon_df.columns if col.startswith(f"{covariate}_")
        ]
        if not candidate_cols:
            candidate_cols = [
                col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
            ]
            if not candidate_cols:
                logger.error(
                    f"Covariate '{covariate}' not found as a direct or one-hot encoded column."
                )
                return {}
            unique_id = engine.properties.dataset.unique_identifier_column
            if (
                unique_id in engine.recon_df.columns
                and unique_id in engine.test_df.columns
            ):
                df = pd.merge(
                    engine.recon_df,
                    engine.test_df[[unique_id] + candidate_cols],
                    on=unique_id,
                )
            else:
                logger.error(
                    f"Unique identifier column not found for covariate '{covariate}'."
                )
                return {}
        else:
            df = engine.recon_df

    latent_cols = [col for col in df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = df[latent_cols].to_numpy()

    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df[covariate]):
            logger.error(
                f"Covariate '{covariate}' is not numeric but task_type='regression'."
            )
            return {}
        y = df[covariate].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=0.0001,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_val = mean_squared_error(y_test, y_pred)
        mae_val = mean_absolute_error(y_test, y_pred)
        r2_val = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        logger.info(
            f"Adversarial MLP (regression) for '{covariate}': MSE={mse_val:.4f}, MAE={mae_val:.4f}, "
            f"R²={r2_val:.4f}, ExplainedVar={explained_var:.4f}"
        )
        return {
            "mse": mse_val,
            "mae": mae_val,
            "r2": r2_val,
            "explained_variance": explained_var,
        }

    if task_type == "classification":
        candidate_cols = [col for col in df.columns if col.startswith(f"{covariate}_")]
        if candidate_cols:
            y = np.argmax(df[candidate_cols].to_numpy(), axis=1)
        else:
            y = df[covariate]
            if pd.api.types.is_numeric_dtype(y):
                y = pd.Categorical(y).codes
            else:
                y = pd.Categorical(y).codes

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error(
                f"Only one class present for covariate '{covariate}' in adversarial classification."
            )
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        if len(np.unique(y_train)) < 2:
            logger.error(
                "Training data in adversarial classifier contains only one class."
            )
            return {}
        model_class = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
        )
        model_class.fit(X_train, y_train)
        y_pred = model_class.predict(X_test)
        y_prob = model_class.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_val = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(
            f"Adversarial MLP (classification) for '{covariate}': Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1={f1_val:.4f}, AUC={(auc if auc is not None else 'N/A')}, Confusion Matrix={cm.tolist()}"
        )
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1_val,
            "auc": auc,
            "confusion_matrix": cm.tolist(),
        }

    logger.error(
        f"Unknown task_type '{task_type}'. Use 'regression' or 'classification'."
    )
    return {}


def calculate_latent_logistic_classification_error(
    engine: Any,
    covariate: str,
    random_state: int = 42,
    max_iter: int = 10000,
    solver: str = "lbfgs",
) -> dict[str, float]:
    """
    Perform logistic regression classification on a one-hot encoded covariate.

    The function detects all columns beginning with the covariate identifier (e.g. "sex_"),
    converts the one-hot representation into categorical labels via argmax, and trains a LogisticRegression
    model using latent features. It then computes additional metrics such as AUC, precision, recall, F1, and confusion matrix.
    If only a single class is found, the method logs an error and returns an error message.
    """
    logger = engine.logger
    candidate_cols = [
        col for col in engine.recon_df.columns if col.startswith(f"{covariate}_")
    ]
    if not candidate_cols:
        candidate_cols = [
            col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
        ]
        if not candidate_cols:
            logger.error(
                f"No one-hot encoded columns found for covariate '{covariate}'."
            )
            return {}
        df = engine.test_df
    else:
        df = engine.recon_df

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = engine.recon_df[latent_cols].to_numpy()
    y_onehot = df[candidate_cols].to_numpy()
    y_cat = np.argmax(y_onehot, axis=1)

    unique_classes = np.unique(y_cat)
    if len(unique_classes) < 2:
        logger.error(
            f"Only one class {unique_classes[0]} found for covariate '{covariate}'."
        )
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=random_state
    )
    if len(np.unique(y_train)) < 2:
        logger.error(
            "Training data contains only one class. Cannot train logistic regression."
        )
        return {}

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_prob = clf.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(
        f"Logistic classification error for '{covariate}' using columns {candidate_cols}: "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, "
        f"AUC={(auc if auc is not None else 'N/A')}, Confusion Matrix={cm.tolist()}"
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def calculate_latent_nonlinear_classification_error(
    engine: Any,
    covariate: str,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Perform nonlinear classification using RandomForestClassifier on a one-hot encoded covariate.

    This function detects one-hot encoded columns for the covariate, converts them to categorical labels,
    and trains a RandomForestClassifier using latent features. It returns metrics including accuracy, AUC,
    precision, recall, F1 score, and the confusion matrix. If only one class exists, it logs and returns an error.
    """
    logger = engine.logger
    candidate_cols = [
        col for col in engine.recon_df.columns if col.startswith(f"{covariate}_")
    ]
    if not candidate_cols:
        candidate_cols = [
            col for col in engine.test_df.columns if col.startswith(f"{covariate}_")
        ]
        if not candidate_cols:
            logger.error(
                f"No one-hot encoded columns found for covariate '{covariate}'."
            )
            return {}
        df = engine.test_df
    else:
        df = engine.recon_df

    latent_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent space columns found with prefix 'z_mean_'.")
        return {}

    X = engine.recon_df[latent_cols].to_numpy()
    y_onehot = df[candidate_cols].to_numpy()
    y_cat = np.argmax(y_onehot, axis=1)

    unique_classes = np.unique(y_cat)
    if len(unique_classes) < 2:
        logger.error(
            f"Only one class {unique_classes[0]} found for covariate '{covariate}'."
        )
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=random_state
    )
    if len(np.unique(y_train)) < 2:
        logger.error(
            "Training data contains only one class. Cannot train RandomForestClassifier."
        )
        return {}

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_prob = clf.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(
        f"Nonlinear classification error for '{covariate}' using columns {candidate_cols}: "
        f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, "
        f"AUC={(auc if auc is not None else 'N/A')}, Confusion Matrix={cm.tolist()}"
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }
