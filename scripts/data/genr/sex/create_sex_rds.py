"""Transform a CSV containing IDC;Sex pairs to a proper RDS."""

import pandas as pd
import pyreadr

# Read the CSV file
df = pd.read_csv("info_s.csv")
# Split the "IDC.sex" column on the semicolon into two new columns
split_cols = df["IDC;sex"].str.split(";", expand=True)
df["idc"] = pd.to_numeric(split_cols[0], errors="coerce")  # Convert to numeric
df["sex"] = split_cols[1].map({"boy": 0, "girl": 1})  # Encode sex: boy -> 0, girl -> 1

# Optionally drop the original "IDC.sex" column if no longer needed
df = df.drop(columns=["IDC;sex"])

# Save the processed DataFrame as an RDS file
pyreadr.write_rds("genr_sex.rds", df)
