#!/bin/bash
#SBATCH --job-name=tsne_explore_job
#SBATCH --time=0:15:00
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/tsne_explore_job_%j.out
#SBATCH --error=slurm_logs/tsne_explore_job_%j.err

# Load the required modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Configure poetry
poetry config installer.max-workers 10

# Install the dependencies
poetry lock --quiet
poetry install --only main --no-interaction --no-ansi --quiet

# Run the t-SNE exploration script
poetry run python scripts/data/genr/merged_data_explore.py
