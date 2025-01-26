#!/bin/bash
#SBATCH --job-name=mt_vae_nm_genr_tuning
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/mt_vae_nm_genr_job_%j.out
#SBATCH --error=slurm_logs/mt_vae_nm_genr_job_%j.err
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1

# Load the required modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Configure poetry
poetry config installer.max-workers 10

# Install the dependencies
poetry lock --quiet
poetry install --only main --no-interaction --no-ansi --quiet

# Run the training script
poetry run python src/main.py --config config_genr.yml --mode tune --debug --verbose --device cuda
