# ----------------------------
# User-defined Parameters
# ----------------------------
id_col <- "EID"                    # Name of the identifier column
thickness_suffix <- "_thickness"   # Suffix for thickness features
area_suffix <- "_area"             # Suffix for area features
dataset_prefix <- "hbn"            # Dataset prefix (e.g., "hbn"); change as needed
exclusion_rate <- 0.1              # Exclusion threshold as a fraction of total features
remove_exact_duplicates <- TRUE    # If TRUE, remove rows that are exactly the same
handle_duplicates <- TRUE          # If TRUE, make duplicate IDs unique and merge duplicate groups

# Build file names based on the dataset prefix
thickness_file <- paste0(dataset_prefix, "_aparc_thickness.rds")
area_file <- paste0(dataset_prefix, "_aparc_area.rds")

# ----------------------------
# STEP 1: Process Thickness Data
# ----------------------------

dat_thick <- readRDS(thickness_file)

# Keep only columns that are the identifier or end with the thickness suffix
keep_cols <- grep(paste0("^(", id_col, "$|.*", thickness_suffix, "$)"), names(dat_thick), value = TRUE)
dat_thick <- dat_thick[, keep_cols]

# Save a copy of the original thickness data (after subsetting) for duplicate flagging
if(remove_exact_duplicates) {
  orig_dat_thick <- dat_thick
}

if(remove_exact_duplicates) {
  n_thick_exact_duplicates <- sum(duplicated(dat_thick))
  if(n_thick_exact_duplicates > 0) {
    cat("Found", n_thick_exact_duplicates, "exact duplicate rows in", thickness_file, ". Removing them...\n")
    dat_thick <- dat_thick[!duplicated(dat_thick), ]
  } else {
    cat("No exact duplicate rows found in", thickness_file, "\n")
  }
}

# Ensure the identifier column is treated as a character vector
dat_thick[[id_col]] <- as.character(dat_thick[[id_col]])

if(handle_duplicates) {
  # Check for duplicated identifiers and, if found, append suffixes to make them unique
  n_duplicates_thick <- sum(duplicated(dat_thick[[id_col]]))
  if(n_duplicates_thick != 0) {
    cat("Found", n_duplicates_thick, "duplicate", id_col, "values in", thickness_file,
        "file. Making them unique by appending suffixes...\n")
    dat_thick[[id_col]] <- ave(dat_thick[[id_col]], dat_thick[[id_col]], FUN = function(x) {
      if(length(x) == 1) x else paste0(x, "_", seq_along(x))
    })
  } else {
    cat("No duplicate", id_col, "values found in", thickness_file, "\n")
  }
} else {
  cat("Not handling duplicates in", thickness_file, "\n")
}

# Number of thickness features (all columns except the identifier)
num_features_thick <- ncol(dat_thick) - 1

# Calculate the lower and upper thresholds for each thickness feature
lower_thick <- rep(NA, num_features_thick)
upper_thick <- rep(NA, num_features_thick)
for(x in 2:ncol(dat_thick)) {
  lower_thick[x - 1] <- mean(dat_thick[, x], na.rm = TRUE) - 2.698 * sd(dat_thick[, x], na.rm = TRUE)
  upper_thick[x - 1] <- mean(dat_thick[, x], na.rm = TRUE) + 2.698 * sd(dat_thick[, x], na.rm = TRUE)
}

# For each subject, count the number of thickness outliers
thick_outlier_counts <- numeric(nrow(dat_thick))
for(i in 1:nrow(dat_thick)) {
  lowind <- which(dat_thick[i, -1] < lower_thick)
  upind  <- which(dat_thick[i, -1] > upper_thick)
  thick_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with the identifier and thickness outlier count
df_thick <- data.frame(dat_thick[, id_col, drop = FALSE],
                       thickness_outliers = thick_outlier_counts)

# --- NEW: Count outliers per thickness feature ---
thickness_feature_counts <- sapply(2:ncol(dat_thick), function(j) {
  sum(dat_thick[, j] < lower_thick[j - 1] | dat_thick[, j] > upper_thick[j - 1], na.rm = TRUE)
})
names(thickness_feature_counts) <- names(dat_thick)[-1]

cat("\nOutlier counts per thickness feature:\n")
print(thickness_feature_counts)

# --- NEW: Identify thickness features with significantly more outliers ---
# Using the IQR method: features with count > Q3 + 1.5 * IQR are flagged.
thick_q1 <- quantile(thickness_feature_counts, 0.25)
thick_q3 <- quantile(thickness_feature_counts, 0.75)
thick_iqr <- thick_q3 - thick_q1
thick_threshold <- thick_q3 + 1.5 * thick_iqr

sig_thick_features <- thickness_feature_counts[thickness_feature_counts > thick_threshold]

cat("\nThickness features with significantly more outliers (IQR threshold =", thick_threshold, "):\n")
if(length(sig_thick_features) > 0) {
  print(sig_thick_features)
} else {
  cat("None found.\n")
}

# ----------------------------
# STEP 2: Process Surface Area Data
# ----------------------------

dat_area <- readRDS(area_file)

# Keep only columns that are the identifier or end with the area suffix
keep_cols <- grep(paste0("^(", id_col, "$|.*", area_suffix, "$)"), names(dat_area), value = TRUE)
dat_area <- dat_area[, keep_cols]

# Save a copy of the original area data (after subsetting) for duplicate flagging
if(remove_exact_duplicates) {
  orig_dat_area <- dat_area
}

if(remove_exact_duplicates) {
  n_area_exact_duplicates <- sum(duplicated(dat_area))
  if(n_area_exact_duplicates > 0) {
    cat("Found", n_area_exact_duplicates, "exact duplicate rows in", area_file, ". Removing them...\n")
    dat_area <- dat_area[!duplicated(dat_area), ]
  } else {
    cat("No exact duplicate rows found in", area_file, "\n")
  }
}

# Ensure the identifier column is treated as a character vector
dat_area[[id_col]] <- as.character(dat_area[[id_col]])

if(handle_duplicates) {
  # Check for duplicated identifiers and, if found, append suffixes to make them unique
  n_duplicates_area <- sum(duplicated(dat_area[[id_col]]))
  if(n_duplicates_area != 0) {
    cat("Found", n_duplicates_area, "duplicate", id_col, "values in", area_file,
        "file. Making them unique by appending suffixes...\n")
    dat_area[[id_col]] <- ave(dat_area[[id_col]], dat_area[[id_col]], FUN = function(x) {
      if(length(x) == 1) x else paste0(x, "_", seq_along(x))
    })
  } else {
    cat("No duplicate", id_col, "values found in", area_file, "\n")
  }
} else {
  cat("Not handling duplicates in", area_file, "\n")
}

# Number of area features (all columns except the identifier)
num_features_area <- ncol(dat_area) - 1

# Calculate the lower and upper thresholds for each area feature
lower_area <- rep(NA, num_features_area)
upper_area <- rep(NA, num_features_area)
for(x in 2:ncol(dat_area)) {
  lower_area[x - 1] <- mean(dat_area[, x], na.rm = TRUE) - 2.698 * sd(dat_area[, x], na.rm = TRUE)
  upper_area[x - 1] <- mean(dat_area[, x], na.rm = TRUE) + 2.698 * sd(dat_area[, x], na.rm = TRUE)
}

# For each subject, count the number of area outliers
area_outlier_counts <- numeric(nrow(dat_area))
for(i in 1:nrow(dat_area)) {
  lowind <- which(dat_area[i, -1] < lower_area)
  upind  <- which(dat_area[i, -1] > upper_area)
  area_outlier_counts[i] <- length(lowind) + length(upind)
}

# Create a data frame with the identifier and area outlier count
df_area <- data.frame(dat_area[, id_col, drop = FALSE],
                      area_outliers = area_outlier_counts)

# --- NEW: Count outliers per area feature ---
area_feature_counts <- sapply(2:ncol(dat_area), function(j) {
  sum(dat_area[, j] < lower_area[j - 1] | dat_area[, j] > upper_area[j - 1], na.rm = TRUE)
})
names(area_feature_counts) <- names(dat_area)[-1]

cat("\nOutlier counts per area feature:\n")
print(area_feature_counts)

# --- NEW: Identify area features with significantly more outliers ---
# Using the IQR method: features with count > Q3 + 1.5 * IQR are flagged.
area_q1 <- quantile(area_feature_counts, 0.25)
area_q3 <- quantile(area_feature_counts, 0.75)
area_iqr <- area_q3 - area_q1
area_threshold <- area_q3 + 1.5 * area_iqr

sig_area_features <- area_feature_counts[area_feature_counts > area_threshold]

cat("\nArea features with significantly more outliers (IQR threshold =", area_threshold, "):\n")
if(length(sig_area_features) > 0) {
  print(sig_area_features)
} else {
  cat("None found.\n")
}

# ----------------------------
# STEP 3: Merge Results and Create Summary
# ----------------------------

# Merge the two data frames by the identifier column.
# (Assumes the same subjects are present in both files.)
df_out <- merge(df_thick, df_area, by = id_col, all = TRUE)

# Replace any NA with 0 (if a subject is missing from one dataset)
df_out[is.na(df_out)] <- 0

# Calculate total outliers per subject
df_out$total_outliers <- df_out$thickness_outliers + df_out$area_outliers

# Total number of features assessed
total_features <- num_features_thick + num_features_area

# Create an "exclude" flag:
# If more than exclusion_rate of all features are outliers, then exclude = 1 (TRUE), else 0 (FALSE)
df_out$exclude <- ifelse(df_out$total_outliers > exclusion_rate * total_features, 1, 0)

# If duplicates are not being handled, check for duplicate identifiers and exit gracefully if found.
if(!handle_duplicates) {
  if(anyDuplicated(df_out[[id_col]]) != 0) {
    cat("Duplicate identifiers found in the merged output and handle_duplicates is set to FALSE.\n")
    stop("Exiting gracefully: Please resolve duplicates or set handle_duplicates = TRUE.")
  }
}

# ----------------------------
# STEP 3.5: Merge Duplicate Groups with Identical Outlier Counts
# ----------------------------
if(handle_duplicates) {
  # Create a base_ID by stripping any appended suffix (e.g., "_2", "_3", etc.)
  df_out$base_ID <- sub("_[0-9]+$", "", df_out[[id_col]])

  # Split the data by base_ID to group originally duplicate entries
  dup_groups <- split(df_out, df_out$base_ID)

  # For each group, if there is more than one row and all have the same total_outliers value,
  # merge them by keeping just one row (and setting the identifier to the base_ID).
  merged_rows <- lapply(dup_groups, function(group) {
    if(nrow(group) > 1 && length(unique(group$total_outliers)) == 1) {
      group[1, id_col] <- group$base_ID[1]
      return(group[1,])
    } else {
      return(group)
    }
  })
  df_out <- do.call(rbind, merged_rows)

  # (Optional) Remove the temporary base_ID column
  df_out$base_ID <- NULL
} else {
  cat("Skipping merging of duplicate groups.\n")
}

# ----------------------------
# STEP 4: Add duplicate_removed Column and Save Summary to RDS
# ----------------------------

# Compute a duplicate flag based on the original datasets.
if(remove_exact_duplicates) {
  # Identify IDs with exact duplicate rows in the thickness data:
  dup_ids_thick <- unique(orig_dat_thick[[id_col]][ duplicated(orig_dat_thick) | duplicated(orig_dat_thick, fromLast = TRUE) ])
  # Identify IDs with exact duplicate rows in the area data:
  dup_ids_area <- unique(orig_dat_area[[id_col]][ duplicated(orig_dat_area) | duplicated(orig_dat_area, fromLast = TRUE) ])
  # Combine the IDs from both datasets.
  dup_ids_all <- unique(c(dup_ids_thick, dup_ids_area))

  # For the final merged output, use the base ID (i.e. remove any suffix) to check for duplicates.
  df_out$duplicate_removed <- ifelse(sub("_[0-9]+$", "", df_out[[id_col]]) %in% dup_ids_all, 1, 0)
} else {
  df_out$duplicate_removed <- 0
}

# Print the number of rows with duplicate_removed set to 1
cat("\nNumber of rows with duplicate_removed set to 1:", sum(df_out$duplicate_removed == 1), "\n")

# Save the final summary including the new duplicate_removed column to RDS
saveRDS(df_out, file = paste0(dataset_prefix, "_qc.rds"))

# ----------------------------
# STEP 5: Additional Summary Reporting
# ----------------------------

# Print the occurrences (frequency) of each total_outliers value
cat("\nOccurrences of total_outliers values:\n")
print(table(df_out$total_outliers))

if(handle_duplicates) {
  # For duplicate identifiers (based on the original value before suffixing),
  # create a base_ID by removing any appended suffix (e.g., "_2", "_3", etc.)
  df_out$base_ID <- sub("_[0-9]+$", "", df_out[[id_col]])

  cat("\nIdentifiers of duplicate groups with differing total_outliers values:\n")
  found_duplicate_diff <- FALSE
  # Split the data by base_ID to identify groups that were originally duplicates
  dup_groups <- split(df_out, df_out$base_ID)
  for (group in dup_groups) {
    if(nrow(group) > 1) {  # More than one entry for this base identifier
      # If the total_outliers values are not all the same, print the group
      if(length(unique(group$total_outliers)) > 1) {
        cat("Base ID:", group$base_ID[1], " -> Full IDs:", paste(group[[id_col]], collapse = ", "),
            " with total_outliers:", paste(group$total_outliers, collapse = ", "), "\n")
        found_duplicate_diff <- TRUE
      }
    }
  }
  if(!found_duplicate_diff) {
    cat("None found.\n")
  }
} else {
  cat("\nSkipping duplicate group reporting since duplicates were not handled.\n")
}

# Finally, print the number of flagged IDs and list them (just the IDs)
flagged_ids <- df_out[[id_col]][df_out$exclude == 1]
cat("\nNumber of flagged IDs:", length(flagged_ids), "\n")
if(length(flagged_ids) > 0) {
  cat("Flagged IDs:", paste(flagged_ids, collapse = ", "), "\n")
}

# ----------------------------
# STEP 6: Report Duplicate Row Removal Summary
# ----------------------------
if(remove_exact_duplicates) {
  cat("\nDuplicate Row Removal Summary:\n")
  cat("Thickness file (", thickness_file, "): Found and removed ", n_thick_exact_duplicates, " exact duplicate rows.\n", sep = "")
  cat("Area file (", area_file, "): Found and removed ", n_area_exact_duplicates, " exact duplicate rows.\n", sep = "")
}
