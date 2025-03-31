"""Script to run PCN-GBR normative model on GENR data."""

import os
import shutil

import numpy as np
import pandas as pd
import pcntoolkit as ptk

# ----- Configuration -----
# Set this flag to True to perform per-site analysis; False to skip it.
PER_SITE_ANALYSIS = True

# File paths for covariates, responses, and batch effects (using PKL files).
covfile = "./input/X_train.pkl"
respfile = "./input/Y_train.pkl"
trbefile = "./input/trbefile.pkl"
tsbefile = "./input/tsbefile.pkl"
testcovfile_path = "./input/X_test.pkl"
testrespfile_path = "./input/Y_test.pkl"

log_dir = "./logs"
output_path = "./output"
roi_dir = "./roi"
outputsuffix = "_genr_pcn_hbr"

# ----- Clear folders before starting -----
folders_to_clear = [log_dir, output_path, roi_dir]
for folder in folders_to_clear:
    if os.path.exists(folder):
        print(f"Clearing folder: {folder}")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
# ----- End Clearing Folders -----

# Get ROI names from the response file.
# We assume the response file (Y_train.pkl) contains one column per ROI.
df_resp = pd.read_pickle(respfile)
rois = df_resp.columns.tolist()
print("ROIs to process:", rois)

# Initialize global DataFrames to collect overall and per-site metrics.
global_all_metrics = pd.DataFrame(columns=["ROI", "MSLL", "EV", "SMSE", "RMSE", "Rho"])
global_site_metrics = pd.DataFrame(
    columns=["ROI", "site", "MSLL", "EV", "SMSE", "RMSE", "Rho"]
)


def save_raw_metrics_as_csv(raw_metrics_, roi_, out_dir):
    """Save each key in the raw_metrics dictionary as a separate CSV file. If the value is a NumPy array, it will be converted into a DataFrame."""
    for key, value in raw_metrics_.items():
        if isinstance(value, np.ndarray):
            df_value = pd.DataFrame(value)
        else:
            # For non-array objects, convert to a DataFrame with one row.
            df_value = pd.DataFrame({key: [value]})
        file_name = os.path.join(out_dir, f"raw_{key}_{roi_}.csv")
        df_value.to_csv(file_name, index=False)
        print(f"Saved raw metric '{key}' for ROI {roi_} to {file_name}")


# Loop over each ROI.
for roi in rois:
    print(f"\nProcessing ROI: {roi}")

    # Create temporary PKL files for the current ROI (for training and test responses)
    temp_resp_train = os.path.join(roi_dir, f"{roi}_train.pkl")
    temp_resp_test = os.path.join(roi_dir, f"{roi}_test.pkl")

    # For training responses: extract only the current ROI column and save.
    df_temp_train = df_resp[[roi]]
    df_temp_train.to_pickle(temp_resp_train)

    # For test responses: load the full test responses file and extract the current ROI.
    df_resp_test = pd.read_pickle(testrespfile_path)
    df_temp_test = df_resp_test[[roi]]
    df_temp_test.to_pickle(temp_resp_test)

    # Estimate the normative model for this ROI.
    # Note: The covariate and batch-effect files remain the same.
    yhat_te, s2_te, nm, Z, metrics_te = ptk.normative.estimate(
        covfile=covfile,
        respfile=temp_resp_train,
        tsbefile=tsbefile,
        trbefile=trbefile,
        inscaler="standardize",
        outscaler="standardize",
        alg="hbr",
        log_path=log_dir,
        binary=True,
        output_path=output_path,
        testcov=testcovfile_path,
        testresp=temp_resp_test,
        outputsuffix=outputsuffix,
        savemodel=False,
        saveoutput=False,
    )

    # Save raw metrics (yhat_te, s2_te, nm, Z) for this ROI as CSV files.
    raw_metrics = {"yhat_te": yhat_te, "s2_te": s2_te, "Z": Z}
    save_raw_metrics_as_csv(raw_metrics, roi, output_path)

    # Store overall (global) metrics for this ROI.
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

    # If per-site analysis is enabled, perform evaluation by site.
    if PER_SITE_ANALYSIS:
        roi_site_metrics = []

        # Load the true test responses for the current ROI.
        temp_test_df = pd.read_pickle(temp_resp_test)
        y_te = temp_test_df.values  # shape (n_samples, 1)
        if y_te.ndim == 1:
            y_te = y_te[:, np.newaxis]

        # Load the training responses for the current ROI.
        temp_train_df = pd.read_pickle(temp_resp_train)
        y_tr = temp_train_df.values
        if y_tr.ndim == 1:
            y_tr = y_tr[:, np.newaxis]

        # Load the test batch effects to determine site membership.
        # We assume that tsbefile contains dummy-encoded site columns (e.g., "site_1", "site_2", etc.)
        df_ts = pd.read_pickle(tsbefile)
        site_cols = [col for col in df_ts.columns if col.startswith("site_")]
        # For each test sample, determine its site by taking the dummy column with maximum value.
        site_series = df_ts[site_cols].idxmax(axis=1)
        unique_sites = site_series.unique()

        # Evaluate metrics for each site.
        for site in unique_sites:
            indices = site_series == site
            y_te_site = y_te[indices]
            yhat_te_site = yhat_te[indices]
            s2_te_site = s2_te[indices]

            # Compute mean and variance of true responses for this site.
            y_mean_te_site = np.mean(y_te_site)
            y_var_te_site = np.var(y_te_site)

            # Evaluate metrics for this site using ptk.normative.evaluate.
            metrics_te_site = ptk.normative.evaluate(
                y_te_site, yhat_te_site, s2_te_site, y_mean_te_site, y_var_te_site
            )

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

        # Save separate CSV file for per-site metrics for this ROI.
        site_csv_file = os.path.join(output_path, f"site_metrics_{roi}.csv")
        site_df.to_csv(site_csv_file, index=False)
        print(f"Saved site metrics for ROI {roi} to {site_csv_file}")
    else:
        print("Per-site analysis is disabled; skipping per-site metrics for this ROI.")

    # Save separate CSV file for overall metrics for this ROI.
    overall_csv_file = os.path.join(output_path, f"overall_metrics_{roi}.csv")
    overall_df.to_csv(overall_csv_file, index=False)
    print(f"Saved overall metrics for ROI {roi} to {overall_csv_file}")

# Save global metrics across all ROIs.
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
