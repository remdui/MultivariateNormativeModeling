"""Data Harmonization with Combat."""

import os
import shutil

import pandas as pd
import pyreadr
from neuroHarmonize import harmonizationLearn

# ----- Configuration -----
harmonize_covars = ["site"]  # Use e.g., ["site", "age"] if needed.
input_rds_file = "../../../data/hbn_aparc_vol_lh_rh.rds"
output_rds_file = "./output/harmonized_hbn_aparc_vol_lh_rh.rds"

# ----- Setup Output Directory -----
output_dir = "./output"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ----- Load Data -----
print(f"Loading data from {input_rds_file} ...")
result = pyreadr.read_r(input_rds_file)
df = list(result.values())[0]
if df is None:
    raise ValueError("No data found in the RDS file.")
print(f"Data loaded. Shape: {df.shape}")

# ----- Extract Covariates and Brain Measure Columns -----
# Extract covariates and convert to category
covars = df[harmonize_covars].copy()
for cov in harmonize_covars:
    covars[cov] = covars[cov].astype("category")
# Rename "site" to "SITE" if present (for Combat)
if "site" in covars.columns:
    covars.rename(columns={"site": "SITE"}, inplace=True)
    print('Renamed covariate "site" to "SITE".')

# Extract brain measure columns (columns starting with 'lh_' or 'rh_')
brain_cols = [
    col for col in df.columns if col.startswith("lh_") or col.startswith("rh_")
]
features = df[brain_cols].copy()
print(f"Identified {len(brain_cols)} brain measure columns.")

# ----- Harmonization -----
# Convert features to numpy array with shape (n_features, n_subjects)
data_array = features.to_numpy()
print("Applying Combat harmonization ...")
model, combat_data_array = harmonizationLearn(data_array, covars)
# Convert the harmonized array back to DataFrame (transpose to original shape)
harmonized_features = pd.DataFrame(
    combat_data_array, columns=features.columns, index=features.index
)

# ----- Replace Brain Measure Columns with Harmonized Data -----
df[brain_cols] = harmonized_features

# ----- Save the Harmonized Data -----
pyreadr.write_rds(output_rds_file, df)
print(f"Saved harmonized data to {output_rds_file}")
print("Combat harmonization completed.")
