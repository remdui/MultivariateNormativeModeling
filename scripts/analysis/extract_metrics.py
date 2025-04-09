"""Script to extract metrics from experiments."""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# === CONFIG ===
base_path = "../../experiments"
keys_to_extract = ["recon_r2", "recon_mse"]


# === Helpers ===
def parse_identifier(identifier):
    """Parse identifier into a dictionary."""
    parts = identifier.split("_")
    parsed = {}
    for part in parts:
        if "-" in part:
            key_, value = part.split("-", 1)
            parsed[key_] = value
        else:
            parsed["embed"] = part
    return parsed


results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [np.nan] * 5)))

for exp_folder in os.listdir(base_path):
    exp_path = os.path.join(base_path, exp_folder)
    if not os.path.isdir(exp_path):
        continue

    info = parse_identifier(exp_folder)
    dataset = info.get("dataset", "unknown")
    covtype = info.get("covtype", "none")
    embed = info.get("embed", "unknown")
    latent = info.get("dim", "NA")
    rep = info.get("rep", None)
    rep_idx = int(rep)

    for task_folder in os.listdir(exp_path):
        if "validate" not in task_folder:
            continue

        metrics_path = os.path.join(
            exp_path, task_folder, "output", "metrics", "metrics.json"
        )
        if not os.path.isfile(metrics_path):
            continue

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        for key in keys_to_extract:
            if key in metrics:
                base_key = (embed, dataset, covtype, latent)
                results[base_key][key][rep_idx] = metrics[key]

rows = []
for (embed, dataset, covtype, latent), metric_data in results.items():
    row = {
        "TypeExperiment": embed,
        "Dataset": dataset,
        "CovType": covtype,
        "Latent": latent,
    }

    for metric, reps in metric_data.items():
        vals = []
        for i in range(5):
            val = reps[i]
            row[f"{metric}_Repeat #{i+1}"] = val
            vals.append(val)
        vals = np.array(vals, dtype=np.float64)
        row[f"{metric}_Average"] = round(np.nanmean(vals), 2)
        row[f"{metric}_Std"] = round(np.nanstd(vals), 2)

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("metrics_summary_grouped.csv", index=False)
