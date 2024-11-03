# Load necessary library
library(tidyverse)

# Number of samples
num_samples <- 3000

# Feature names from a realistic FreeSurfer atlas
feature_names <- c(
  "Left-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Left-Cerebellum-White-Matter", "Left-Cerebellum-Cortex", "Left-Thalamus-Proper",
  "Left-Caudate", "Left-Putamen", "Left-Pallidum", "3rd-Ventricle", "4th-Ventricle",
  "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "CSF", "Left-Accumbens-area",
  "Left-VentralDC", "Left-vessel", "Left-choroid-plexus", "Right-Lateral-Ventricle", "Right-Inf-Lat-Vent",
  "Right-Cerebellum-White-Matter", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen",
  "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area", "Right-VentralDC",
  "Right-vessel", "Right-choroid-plexus", "5th-Ventricle", "WM-hypointensities", "Left-WM-hypointensities",
  "Right-WM-hypointensities", "non-WM-hypointensities", "Left-non-WM-hypointensities", "Right-non-WM-hypointensities",
  "Optic-Chiasm", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior",
  "CC_Anterior", "Cerebral-White-Matter", "Cerebral-Cortex", "Lateral-Ventricle", "Inf-Lat-Vent",
  "Cerebellum-White-Matter", "Cerebellum-Cortex", "Thalamus-Proper", "Caudate", "Putamen",
  "Pallidum", "Hippocampus", "Amygdala", "Accumbens-area", "VentralDC",
  "vessel", "choroid-plexus", "CSF-Mask", "Cerebral-White-Matter-Mask", "Cerebral-Cortex-Mask",
  "Lateral-Ventricle-Mask", "Inf-Lat-Vent-Mask", "Cerebellum-White-Matter-Mask", "Cerebellum-Cortex-Mask", "Thalamus-Proper-Mask",
  "Caudate-Mask", "Putamen-Mask", "Pallidum-Mask", "Hippocampus-Mask", "Amygdala-Mask",
  "Accumbens-area-Mask", "VentralDC-Mask", "vessel-Mask", "choroid-plexus-Mask", "Optic-Chiasm-Mask",
  "CC_Posterior-Mask", "CC_Mid_Posterior-Mask", "CC_Central-Mask", "CC_Mid_Anterior-Mask", "CC_Anterior-Mask",
  "Brain-Stem-Mask", "3rd-Ventricle-Mask", "4th-Ventricle-Mask", "5th-Ventricle-Mask", "WM-hypointensities-Mask",
  "non-WM-hypointensities-Mask", "Left-Lesion", "Right-Lesion", "Left-Cerebellar-Lesion", "Right-Cerebellar-Lesion",
  "Brain-Stem-Lesion", "Left-Accumbens", "Right-Accumbens", "Left-Amygdala", "Right-Amygdala"
)

# Generate random continuous values for features
data <- matrix(rnorm(num_samples * length(feature_names), mean = 2.5, sd = 0.5), nrow = num_samples, ncol = length(feature_names))

# Generate binary values for gender (0 for female, 1 for male)
gender <- sample(0:1, num_samples, replace = TRUE, prob = c(0.5, 0.5))

# Generate age values (mean age 13, sd 3, bounded between 7 and 19)
age <- pmax(pmin(round(rnorm(num_samples, mean = 13, sd = 3)), 19), 7)

# Combine all features into a single data frame
df <- as.data.frame(data)
df$age <- age
df$gender <- gender

# Rename columns
colnames(df) <- c(feature_names, "age", "gender")

# Save to RDS file
saveRDS(df, "generated_data.rds")
