"""Create PCN input files for the HBN dataset.

Follows:
https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo/blob/main/tutorials/BLR_protocol/BLR_normativemodel_protocol.ipynb
and
https://pcntoolkit.readthedocs.io/en/latest/pages/HBR_NormativeModel_FCONdata_Tutorial.html
"""

import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility (optional)
np.random.seed(42)

# Set saving format: 'pkl' or 'csv'
SAVE_FORMAT = "pkl"  # Change to 'pkl' to save as PKL files instead


def print_stats(df_, name):
    """Print basic stats for a DataFrame."""
    print(f"Stats for {name}:")
    print(f" - Shape: {df_.shape}")
    print(f" - Columns: {list(df_.columns)}")
    print(df_.head(), "\n")


def save_df(df_, filepath, header=True):
    """Save DataFrame as either pickle or csv based on SAVE_FORMAT.

    For CSV, the header row is written based on the 'header' flag.
    """
    if SAVE_FORMAT == "pkl":
        with open(filepath, "wb") as f:
            pickle.dump(df_, f)
    elif SAVE_FORMAT == "csv":
        df_.to_csv(filepath, index=False, header=header)
    else:
        raise ValueError("Unsupported SAVE_FORMAT. Use 'pkl' or 'csv'.")


# Determine file extension based on SAVE_FORMAT
ext = SAVE_FORMAT

# Path to the RDS file
datapath = "../../../../../data/hbn_aparc_vol_lh_rh.rds"
print(f"Loading data from {datapath}...")

# Read the RDS file using pyreadr
result = pyreadr.read_r(
    datapath
)  # returns a dict with keys as object names in the file
# Assume the first object is the DataFrame we need
df = list(result.values())[0]
if df is None:
    raise ValueError("No data found in the RDS file.")
print(f"Data loaded. DataFrame shape: {df.shape}")
print("Columns in the loaded DataFrame:", df.columns.tolist(), "\n")

# Drop unwanted columns 'row_id' and 'EID' (if they exist)
df = df.drop(columns=["row_id", "EID"], errors="ignore")
print("After dropping 'row_id' and 'EID':")
print_stats(df, "DataFrame after dropping columns")

# Save a copy of the original 'site' column for stratification
orig_site = df["site"].copy()

# Dummy encode categorical variables for covariates: 'site' and 'sex'
df = pd.get_dummies(df, columns=["site", "sex"])
dummy_cols = [
    col for col in df.columns if col.startswith("site_") or col.startswith("sex_")
]
print("After dummy encoding 'site' and 'sex':")
print("Dummy variable columns added:", dummy_cols)
print(df.head(), "\n")

# Identify measurement columns: all columns starting with 'lh_' or 'rh_'
measurement_cols = [
    col for col in df.columns if col.startswith("lh_") or col.startswith("rh_")
]
print("Identified measurement columns:")
print(measurement_cols, "\n")

# ---------------------------
# Step 1: Split into training and testing sets using train_test_split with stratification
# ---------------------------
# Stratify based on the original 'site' column (before dummy encoding)
df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=orig_site, random_state=42
)

print("Data split into training and testing sets:")
print(f" - Training set shape: {df_train.shape}")
print(f" - Testing set shape: {df_test.shape}\n")

# Print sample sizes per site using the dummy columns
print("Sample size per site (dummy encoded):")
for col in sorted([c for c in dummy_cols if c.startswith("site_")]):
    n_train = df_train[col].sum()
    n_test = df_test[col].sum()
    print(f"{col}: Train = {int(n_train)}, Test = {int(n_test)}")
print("\n")

# ---------------------------
# Step 2: Prepare HBR inputs
# ---------------------------
# Covariate: include age plus the dummy columns for site and sex.
covariate_cols = ["age"] + dummy_cols
X_train = df_train[covariate_cols].to_numpy(dtype=float)
X_test = df_test[covariate_cols].to_numpy(dtype=float)

print("Covariate processing (age, site, sex):")
print("X_train sample (first 5):", X_train[:5])
print("X_test sample (first 5):", X_test[:5], "\n")

# Measurements: all lh and rh features
Y_train = df_train[measurement_cols].to_numpy(dtype=float)
Y_test = df_test[measurement_cols].to_numpy(dtype=float)
print("Measurements processing:")
print("Y_train shape:", Y_train.shape)
print("Y_train sample (first 5 rows):")
print(pd.DataFrame(Y_train, columns=measurement_cols).head(), "\n")
print("Y_test shape:", Y_test.shape)
print("Y_test sample (first 5 rows):")
print(pd.DataFrame(Y_test, columns=measurement_cols).head(), "\n")

# ---------------------------
# Clear the output directory before saving
# ---------------------------
output_dir = os.path.join("..", "input")
if os.path.exists(output_dir):
    print(f"Clearing existing files in {output_dir}...")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs(output_dir, exist_ok=True)
print(f"Saving processed files to {output_dir}...\n")

# ---------------------------
# Save files in the output directory
# ---------------------------
# Save X_train (do not save header)
df_X_train = pd.DataFrame(X_train, columns=covariate_cols)
x_train_path = os.path.join(output_dir, f"X_train.{ext}")
save_df(df_X_train, x_train_path, header=False)
print_stats(df_X_train, f"X_train.{ext}")

# Save Y_train (save header)
df_Y_train = pd.DataFrame(Y_train, columns=measurement_cols)
y_train_path = os.path.join(output_dir, f"Y_train.{ext}")
save_df(df_Y_train, y_train_path, header=True)
print_stats(df_Y_train, f"Y_train.{ext}")

# Save X_test (do not save header)
df_X_test = pd.DataFrame(X_test, columns=covariate_cols)
x_test_path = os.path.join(output_dir, f"X_test.{ext}")
save_df(df_X_test, x_test_path, header=False)
print_stats(df_X_test, f"X_test.{ext}")

# Save Y_test (save header)
df_Y_test = pd.DataFrame(Y_test, columns=measurement_cols)
y_test_path = os.path.join(output_dir, f"Y_test.{ext}")
save_df(df_Y_test, y_test_path, header=True)
print_stats(df_Y_test, f"Y_test.{ext}")

print("Data processing completed. All files are saved in:", output_dir)
