"""Generate t-SNE plot for 'thickavg' features colored by age bins."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
from sklearn.manifold import TSNE

exp_name = "thickavg_rh"

# Path to the newly created file
file_path = "./data/gen_r_aparc_wave_f05_f09_f13.rds"

# Output directory for the plot
output_dir = "./output/data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"tsne_surfarea_age_bins_{exp_name}.png")

data = pyreadr.read_r(file_path)[None]  # Load as pandas DataFrame

# Print column names
print(f"Column names in {file_path}:")
print(data.columns.tolist())

# Select features based on filter
features = data.filter(like="thickavg")
features = features[[col for col in features.columns if col.startswith("rh_")]]

# Handle 'object' columns
object_cols = features.select_dtypes(include=["object"])
print(f"Object columns detected: {object_cols.columns.tolist()}")
for col in object_cols.columns:
    try:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    except ValueError as e:
        print(f"Error converting column {col} to numeric: {e}")

# Keep only numeric columns and drop rows with NaN values
numeric_features = features.select_dtypes(include=[np.number]).dropna()

# Check if numeric_features is empty
if numeric_features.empty:
    raise ValueError("No valid numeric 'vol' features found after preprocessing.")

# Ensure all values are float64
numeric_features = numeric_features.astype(np.float64)

# Print the shape of numeric features
print(f"Shape of numeric features: {numeric_features.shape}")

# Verify that the 'age' column exists and align it
if "age" not in data.columns:
    raise ValueError("'age' column not found in the dataset.")
ages = data["age"].dropna().astype(float)

# Align ages and numeric_features to avoid mismatches
aligned_data = pd.concat([numeric_features, ages], axis=1, join="inner")
numeric_features = aligned_data[numeric_features.columns]
ages = aligned_data["age"]

# Define age bins and labels
bins = [0, 7, 11, 15, np.inf]
labels = ["0-7", "7-11", "11-15", "15+"]
age_bins = pd.cut(ages, bins=bins, labels=labels)

# Run t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=1000)
tsne_results = tsne.fit_transform(numeric_features)

# Add the t-SNE results and age bins back to a DataFrame
tsne_df = pd.DataFrame(tsne_results, columns=["tSNE1", "tSNE2"])
tsne_df["age_bin"] = age_bins.reset_index(drop=True)

# Verify the t-SNE output
if tsne_df[["tSNE1", "tSNE2"]].isnull().any().any():
    raise ValueError("t-SNE output contains NaN values. Check the input data.")

# Plot the t-SNE results, coloring by age bins
plt.figure(figsize=(10, 8))
for label in labels:
    subset = tsne_df[tsne_df["age_bin"] == label]
    plt.scatter(subset["tSNE1"], subset["tSNE2"], label=label, alpha=0.7)

plt.legend(title="Age Bin")
plt.title(f"t-SNE Visualization ({exp_name} Features) Colored by Age Bins")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# Save the plot to a file
plt.savefig(output_file, dpi=300)
print(f"t-SNE plot saved to {output_file}")
