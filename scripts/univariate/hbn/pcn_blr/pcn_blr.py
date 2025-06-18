"""Script to run PCN-BLR normative model on HBN data."""

import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pcntoolkit as ptk
from pcntoolkit.util.utils import create_bspline_basis

# ----- Configuration -----
PER_SITE_ANALYSIS = False

# File paths for covariates and responses (using PKL files).
# The covariate files include age (in the first column), one-hot encoded site, and sex.
covfile = "./input/X_train.pkl"
respfile = "./input/Y_train.pkl"
testcovfile_path = "./input/X_test.pkl"
testrespfile_path = "./input/Y_test.pkl"

log_dir = "./logs"
output_path = "./output"
roi_dir = "./roi"
outputsuffix = "_hbn_pcn_blr"

# ----- Clear folders before starting -----
folders_to_clear = [log_dir, output_path, roi_dir]
for folder in folders_to_clear:
    if os.path.exists(folder):
        print(f"Clearing folder: {folder}")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
# ----- End Clearing Folders -----


def print_stats(df, name):
    """Prints basic statistics for a given DataFrame."""
    print(f"Stats for {name}:")
    print(f" - Shape: {df.shape}")
    print(f" - Columns: {list(df.columns)}")
    print(df.head(), "\n")


def save_df(df, filepath, header=True):
    """Saves a DataFrame to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(df, f)


def save_raw_metrics_as_csv(raw_metrics_, roi_, out_dir):
    """Saves raw metrics as CSV files."""
    for key, value in raw_metrics_.items():
        if isinstance(value, np.ndarray):
            df_value = pd.DataFrame(value)
        else:
            df_value = pd.DataFrame({key: [value]})
        file_name = os.path.join(out_dir, f"raw_{key}_{roi_}.csv")
        df_value.to_csv(file_name, index=False)
        print(f"Saved raw metric '{key}' for ROI {roi_} to {file_name}")


# Set up the B-spline basis.
xmin = 4
xmax = 25
B = create_bspline_basis(xmin, xmax)

# ----- Load Global Covariates & Responses -----
df_cov_train_orig = pd.read_pickle(covfile)
df_cov_test_orig = pd.read_pickle(testcovfile_path)
df_resp = pd.read_pickle(respfile)
rois = df_resp.columns.tolist()
print("ROIs to process:", rois)

global_all_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho"])
global_site_metrics = pd.DataFrame(
    columns=["ROI", "site", "MSLL", "EV", "SMSE", "RMSE", "Rho"]
)

# Loop over each ROI.
for roi in rois:
    print(f"\nProcessing ROI: {roi}")

    # ---------------------
    # Prepare Response Files (per ROI)
    # ---------------------
    temp_resp_train = os.path.join(roi_dir, f"{roi}_train.pkl")
    temp_resp_test = os.path.join(roi_dir, f"{roi}_test.pkl")
    df_temp_train = df_resp[[roi]]
    df_temp_train.to_pickle(temp_resp_train)
    df_resp_test = pd.read_pickle(testrespfile_path)
    df_temp_test = df_resp_test[[roi]]
    df_temp_test.to_pickle(temp_resp_test)

    # ---------------------
    # Prepare Covariate Files with B-Spline Basis Expansion (per ROI)
    # ---------------------
    # Copy the global covariate DataFrames (they include unlabeled age, site, and sex)
    cov_train = df_cov_train_orig.copy()
    cov_test = df_cov_test_orig.copy()

    # Add an intercept column.
    cov_train["intercept"] = 1
    cov_test["intercept"] = 1

    # Perform B-spline basis expansion on the first column (assumed to be age).
    # We call B(x) for each x individually.
    Phi_train = np.array([B(x) for x in cov_train.iloc[:, 0].values])
    Phi_test = np.array([B(x) for x in cov_test.iloc[:, 0].values])

    n_basis = Phi_train.shape[1]
    bspline_cols = [f"bspline_{i}" for i in range(n_basis)]

    # Convert to DataFrames.
    df_Phi_train = pd.DataFrame(Phi_train, columns=bspline_cols, index=cov_train.index)
    df_Phi_test = pd.DataFrame(Phi_test, columns=bspline_cols, index=cov_test.index)

    # Concatenate the original covariates with the new B-spline features.
    cov_train_exp = pd.concat([cov_train, df_Phi_train], axis=1)
    cov_test_exp = pd.concat([cov_test, df_Phi_test], axis=1)

    # Save the expanded covariate files as DataFrames (PKL).
    temp_cov_train = os.path.join(roi_dir, f"cov_bspline_train_{roi}.pkl")
    temp_cov_test = os.path.join(roi_dir, f"cov_bspline_test_{roi}.pkl")
    cov_train_exp.to_pickle(temp_cov_train)
    cov_test_exp.to_pickle(temp_cov_test)
    print(
        f"Saved expanded covariates for ROI {roi} to {temp_cov_train} and {temp_cov_test}"
    )

    # ---------------------
    # Estimate the Normative Model using BLR with Expanded Covariates
    yhat_te, s2_te, nm, Z, metrics_te = ptk.normative.estimate(
        covfile=temp_cov_train,
        respfile=temp_resp_train,
        testcov=temp_cov_test,
        testresp=temp_resp_test,
        alg="blr",
        optimizer="powell",
        standardize=False,
        log_path=log_dir,
        output_path=output_path,
        outputsuffix=outputsuffix,
        savemodel=False,
        saveoutput=False,
    )
    # ---------------------

    # Save raw metrics as CSV files.
    raw_metrics = {"yhat_te": yhat_te, "s2_te": s2_te, "Z": Z}
    save_raw_metrics_as_csv(raw_metrics, roi, output_path)

    # Store overall metrics.
    overall_metrics = {
        "ROI": roi,
        "MSLL": metrics_te["MSLL"][0],
        "EV": metrics_te["EXPV"][0],
        "SMSE": metrics_te["SMSE"][0],
        "RMSE": metrics_te["RMSE"][0],
        "Rho": metrics_te["Rho"][0],
    }
    print("Overall metrics:", overall_metrics)
    overall_df = pd.DataFrame([overall_metrics])
    global_all_metrics = pd.concat([global_all_metrics, overall_df], ignore_index=True)

    # ---------------------
    # Per-Site Analysis (Optional)
    # ---------------------
    if PER_SITE_ANALYSIS:
        roi_site_metrics = []
        # For per-site evaluation, load the expanded test covariates.
        df_cov_test_exp = pd.read_pickle(temp_cov_test)
        # Assume the original covariate file (before expansion) contained dummy-encoded site columns.
        df_cov_test_orig = pd.read_pickle(testcovfile_path)
        site_cols = [col for col in df_cov_test_orig.columns if col.startswith("site_")]
        site_series = df_cov_test_orig[site_cols].idxmax(axis=1)
        unique_sites = site_series.unique()

        temp_test_df = pd.read_pickle(temp_resp_test)
        y_te = temp_test_df.to_numpy()
        if y_te.ndim == 1:
            y_te = y_te[:, np.newaxis]

        for site in unique_sites:
            indices = site_series == site
            y_te_site = y_te[indices]
            yhat_te_site = yhat_te[indices]
            s2_te_site = s2_te[indices]

            y_mean_te_site = np.mean(y_te_site)
            y_var_te_site = np.var(y_te_site)

            metrics_te_site = ptk.normative.evaluate(
                y_te_site, yhat_te_site, s2_te_site, y_mean_te_site, y_var_te_site
            )
            print(metrics_te_site["SMSE"])
            site_metric_row = {
                "ROI": roi,
                "site": site,
                "MSLL": metrics_te_site["MSLL"][0],
                "EV": metrics_te_site["EXPV"][0],
                "SMSE": metrics_te_site["SMSE"][0],
                "RMSE": metrics_te_site["RMSE"][0],
                "Rho": metrics_te_site["Rho"][0],
            }
            print(f"Metrics for site {site}:", site_metric_row)
            roi_site_metrics.append(site_metric_row)

        site_df = pd.DataFrame(roi_site_metrics)
        global_site_metrics = pd.concat(
            [global_site_metrics, site_df], ignore_index=True
        )
        site_csv_file = os.path.join(output_path, f"site_metrics_{roi}.csv")
        site_df.to_csv(site_csv_file, index=False)
        print(f"Saved site metrics for ROI {roi} to {site_csv_file}")
    else:
        print("Per-site analysis is disabled; skipping per-site metrics for this ROI.")

    overall_csv_file = os.path.join(output_path, f"overall_metrics_{roi}.csv")
    overall_df.to_csv(overall_csv_file, index=False)
    print(f"Saved overall metrics for ROI {roi} to {overall_csv_file}")

global_all_metrics.to_csv(
    os.path.join(output_path, "overall_metrics_all.csv"), index=False
)
if PER_SITE_ANALYSIS:
    global_site_metrics.to_csv(
        os.path.join(output_path, "site_metrics_all.csv"), index=False
    )
    print("\nGlobal per-site metrics saved to 'site_metrics_all.csv'.")
else:
    print("\nPer-site analysis was disabled; no global site metrics file generated.")
print("Global overall metrics saved to 'overall_metrics_all.csv'.")

print("\nProcessing completed.")
