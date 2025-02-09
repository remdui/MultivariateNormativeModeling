#!/bin/bash
#SBATCH --job-name=mt_vae_nm_genr
#SBATCH --time=0:30:00
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/mt_vae_nm_genr_job_%j.out
#SBATCH --error=slurm_logs/mt_vae_nm_genr_job_%j.err

# Load the required modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Configure poetry
poetry config installer.max-workers 10

# Install the dependencies
poetry lock --quiet
poetry install --only main --no-interaction --no-ansi --quiet

# Run the training script
poetry run python src/main.py --config config_genr.yml --mode train --debug --verbose --device cpu
