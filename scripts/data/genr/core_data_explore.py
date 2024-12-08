"""Explore the core data file."""

import pyreadr

# Path to the core data file
core_file = "genr_mri_core_data_20231204.rds"

# Load core data
core_data = pyreadr.read_r(core_file)[None]  # Load as pandas DataFrame

# Print the column names
print("Column names in core data:")
print(core_data.columns.tolist())
