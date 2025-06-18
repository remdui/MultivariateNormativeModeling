import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
np.random.seed(42)
SAVE_FORMAT = "pkl"  # 'pkl' or 'csv'
DATA_RDS = "../../../../../data/hbn_aparc_vol.rds"
OUTPUT_DIR = os.path.join("..", "input")
# ---------------------------------------------------------------------


def print_stats(df_: pd.DataFrame, name: str):
    print(f"{name}: shape={df_.shape}, cols={list(df_.columns)}")
    print(df_.head(), "\n")


def save_df(df_: pd.DataFrame, path: str, header: bool = True):
    if SAVE_FORMAT == "pkl":
        with open(path, "wb") as f:
            pickle.dump(df_, f)
    elif SAVE_FORMAT == "csv":
        df_.to_csv(path, index=False, header=header)
    else:
        raise ValueError("SAVE_FORMAT must be 'pkl' or 'csv'")


ext = SAVE_FORMAT  # file extension

# ---------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------
print(f"Loading {DATA_RDS} …")
raw = list(pyreadr.read_r(DATA_RDS).values())[0]  # first object
# drop unwanted columns
df = raw.drop(columns=["row_id", "EID"], errors="ignore")
print_stats(df, "raw df")

# ---------------------------------------------------------------------
# 2. Encode site & sex
# ---------------------------------------------------------------------
# keep string version for stratification
df["site_str"] = df["site"].astype(str)
# site as categorical integer code
df["site_id"] = df["site"].astype("category").cat.codes
# sex as integer code (F=0, M=1)
df["sex_id"] = df["sex"].astype(int)

batch_cols = ["site_id", "sex_id"]  # columns for random effects
print_stats(df[batch_cols + ["site_str"]], "encoded site/sex sample")

# ---------------------------------------------------------------------
# 3. Identify measurements
# ---------------------------------------------------------------------
measurement_cols = [c for c in df.columns if c.endswith("_vol")]
print(f"found {len(measurement_cols)} measurement cols\n")

# ---------------------------------------------------------------------
# 4. Train/test split (stratified by site)
# ---------------------------------------------------------------------
df_train, df_test = train_test_split(
    df, test_size=0.20, stratify=df["site_str"], random_state=42
)

print("train/test sizes per site:")
for s, _ in df.groupby("site_str"):
    n_train = (df_train["site_str"] == s).sum()
    n_test = (df_test["site_str"] == s).sum()
    print(f"  {s}: train={n_train:3d}   test={n_test:3d}")
print()

# ---------------------------------------------------------------------
# 5. Build matrices and standardize
# ---------------------------------------------------------------------
# Age: z-score standardization based on training set
age_scaler = StandardScaler()
X_train = age_scaler.fit_transform(df_train[["age"]].to_numpy())
X_test = age_scaler.transform(df_test[["age"]].to_numpy())

# Brain measures: z-score standardization based on training set
brain_scaler = StandardScaler()
Y_train = brain_scaler.fit_transform(df_train[measurement_cols].to_numpy())
Y_test = brain_scaler.transform(df_test[measurement_cols].to_numpy())

# Batch effects (site_id, sex_id)
BE_train = df_train[batch_cols].to_numpy(dtype=int)
BE_test = df_test[batch_cols].to_numpy(dtype=int)

# Quick sanity prints
print_stats(pd.DataFrame(X_train, columns=["age_zscore"]), "X_train head")
print_stats(pd.DataFrame(BE_train, columns=batch_cols), "BE_train head")
print_stats(pd.DataFrame(Y_train, columns=measurement_cols).head(), "Y_train head")
print()

# ---------------------------------------------------------------------
# 6. Write to disk
# ---------------------------------------------------------------------
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def dump(name, arr_or_df, header):
    # convert array to DataFrame if needed
    df_out = (
        arr_or_df if isinstance(arr_or_df, pd.DataFrame) else pd.DataFrame(arr_or_df)
    )
    save_df(df_out, os.path.join(OUTPUT_DIR, f"{name}.{ext}"), header=header)
    print_stats(df_out, f"{name}.{ext}")


# save matrices and data
dump("X_train", X_train, header=False)
dump("Y_train", pd.DataFrame(Y_train, columns=measurement_cols), header=True)
dump("trbefile", BE_train, header=False)

dump("X_test", X_test, header=False)
dump("Y_test", pd.DataFrame(Y_test, columns=measurement_cols), header=True)
dump("tsbefile", BE_test, header=False)

print(f"✓ Finished – files saved to {OUTPUT_DIR}")
