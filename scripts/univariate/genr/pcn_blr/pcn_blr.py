import os
import shutil

import numpy as np
import pandas as pd
import pcntoolkit as ptk
from pcntoolkit.util.utils import create_bspline_basis

"""Script to run PCN-BLR normative model on HBN data with B-spline basis expansion."""

# ----- Configuration -----
PER_SITE_ANALYSIS = False

# File paths for covariates and responses (using PKL files).
covfile = "./input/X_train.pkl"
respfile = "./input/Y_train.pkl"
testcovfile_path = "./input/X_test.pkl"
testrespfile_path = "./input/Y_test.pkl"

# Directories for logs, outputs, and per-ROI data
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


# Set up the B-spline basis
xmin = 5
xmax = 17
B = create_bspline_basis(xmin, xmax)

# ----- Load Global Covariates & Responses -----
df_cov_train_orig = pd.read_pickle(covfile)
df_cov_test_orig = pd.read_pickle(testcovfile_path)
df_resp = pd.read_pickle(respfile)
rois = df_resp.columns.tolist()
print("ROIs to process:", rois)

global_all_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho"])
global_site_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho"])

# Loop over each ROI.
for roi in rois:
    print(f"\nProcessing ROI: {roi}")

    # Create ROI-specific directory
    roi_path = os.path.join(roi_dir, roi)
    os.makedirs(roi_path, exist_ok=True)

    # ---------------------
    # Prepare Response Files (per ROI)
    # ---------------------
    df_temp_train = df_resp[[roi]]
    df_temp_test = pd.read_pickle(testrespfile_path)[[roi]]

    # Save responses as whitespace-delimited text
    resp_tr_txt = os.path.join(roi_path, "resp_tr.txt")
    resp_te_txt = os.path.join(roi_path, "resp_te.txt")
    np.savetxt(resp_tr_txt, df_temp_train.values, fmt="%g")
    np.savetxt(resp_te_txt, df_temp_test.values, fmt="%g")

    # ---------------------
    # Prepare Covariate Files with B-Spline Basis Expansion (per ROI)
    # ---------------------
    cov_train = df_cov_train_orig.copy()
    cov_test = df_cov_test_orig.copy()

    # Explicitly grab the age column and add intercept
    age_train = cov_train["age"].values
    age_test = cov_test["age"].values
    cov_train["intercept"] = 1
    cov_test["intercept"] = 1

    # Build the B-spline features
    Phi_train = np.vstack([B(a) for a in age_train])
    Phi_test = np.vstack([B(a) for a in age_test])
    n_basis = Phi_train.shape[1]
    bs_cols = [f"bspline_{i}" for i in range(n_basis)]
    df_bs_tr = pd.DataFrame(Phi_train, columns=bs_cols, index=cov_train.index)
    df_bs_te = pd.DataFrame(Phi_test, columns=bs_cols, index=cov_test.index)

    # Concatenate the expanded covariates
    cov_train_exp = pd.concat([cov_train, df_bs_tr], axis=1)
    cov_test_exp = pd.concat([cov_test, df_bs_te], axis=1)

    # Save expanded covariates as whitespace-delimited text
    cov_tr_txt = os.path.join(roi_path, "cov_bspline_tr.txt")
    cov_te_txt = os.path.join(roi_path, "cov_bspline_te.txt")
    np.savetxt(cov_tr_txt, cov_train_exp.values, fmt="%g")
    np.savetxt(cov_te_txt, cov_test_exp.values, fmt="%g")
    print(f"Saved expanded covariates for ROI {roi} to {cov_tr_txt} and {cov_te_txt}")

    # ---------------------
    # Estimate the Normative Model using BLR with Expanded Covariates
    # ---------------------
    yhat_te, s2_te, nm, Z, metrics_te = ptk.normative.estimate(
        covfile=cov_tr_txt,
        respfile=resp_tr_txt,
        testcov=cov_te_txt,
        testresp=resp_te_txt,
        alg="blr",
        optimizer="powell",
        standardize=False,
        log_path=log_dir,
        output_path=output_path,
        outputsuffix=outputsuffix,
        savemodel=False,
        saveoutput=False,
    )

    # Save raw metrics as CSV files
    raw_metrics = {"yhat_te": yhat_te, "s2_te": s2_te, "Z": Z}
    save_raw_metrics_as_csv(raw_metrics, roi, output_path)

    # Store overall metrics
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
        df_cov_test_exp = cov_test_exp.copy()
        site_cols = [c for c in df_cov_test_orig.columns if c.startswith("site_")]
        site_series = df_cov_test_orig[site_cols].idxmax(axis=1)
        y_te = df_temp_test.values
        yhat = yhat_te
        s2 = s2_te
        roi_site_metrics = []
        for site in site_series.unique():
            idx = (site_series == site).values
            y_s = y_te[idx]
            yhat_s = yhat[idx]
            s2_s = s2[idx]
            metrics_site = ptk.normative.evaluate(
                y_s, yhat_s, s2_s, y_s.mean(), y_s.var()
            )
            row = {
                "ROI": roi,
                "site": site,
                "MSLL": metrics_site["MSLL"][0],
                "EV": metrics_site["EXPV"][0],
                "SMSE": metrics_site["SMSE"][0],
                "RMSE": metrics_site["RMSE"][0],
                "Rho": metrics_site["Rho"][0],
            }
            roi_site_metrics.append(row)
        site_df = pd.DataFrame(roi_site_metrics)
        global_site_metrics = pd.concat(
            [global_site_metrics, site_df], ignore_index=True
        )
        site_csv_file = os.path.join(output_path, f"site_metrics_{roi}.csv")
        site_df.to_csv(site_csv_file, index=False)
        print(f"Saved site metrics for ROI {roi} to {site_csv_file}")
    else:
        print("Per-site analysis is disabled; skipping per-site metrics for this ROI.")

    # Save overall metrics per ROI
    overall_csv = os.path.join(output_path, f"overall_metrics_{roi}.csv")
    overall_df.to_csv(overall_csv, index=False)
    print(f"Saved overall metrics for ROI {roi} to {overall_csv}")

# Save global metrics
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
print("\nGlobal overall metrics saved to 'overall_metrics_all.csv'.")
print("\nProcessing completed.")
