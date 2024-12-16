"""Load and explore an RDS file using pyreadr."""

import sys

import pyreadr

# Check if file path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

# File path to the .rds file
file_path = sys.argv[1]

# Load the RDS file using pyreadr
try:
    df = pyreadr.read_r(file_path)[0]

    print(f"Data successfully loaded. Dataframe shape: {df.shape}")
except OSError as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Display basic information about the dataframe
print("\nColumns in the dataframe:")
print(df.columns)
print("\nSample data:")
print(df.head())

# Summary statistics of the numeric columns
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)
