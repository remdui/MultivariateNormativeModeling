"""Script to generate dummy data for testing purposes."""

import numpy as np
import pandas as pd

# Number of samples
num_samples = 3000

# Generate random continuous values for features f1 to f100
data = np.random.rand(num_samples, 100)

# Generate binary values for sex (0 or 1)
sex = np.random.randint(0, 2, size=(num_samples, 1))

# Generate age values (7 to 19)
age = np.random.randint(7, 20, size=(num_samples, 1))

# Combine all features into a single array
combined_data = np.hstack((data, sex, age))

# Create a DataFrame
columns = [f"f{i}" for i in range(1, 101)] + ["sex", "age"]
df = pd.DataFrame(combined_data, columns=columns)

# Save to CSV
df.to_csv("generated_data.csv", index=False)
