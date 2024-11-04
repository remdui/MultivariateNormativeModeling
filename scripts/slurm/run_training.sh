#!/bin/bash
#SBATCH --job-name=mt_vae_training_job
#SBATCH --time=0:10:00
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/mt_vae_training_job_%j.out
#SBATCH --error=logs/mt_vae_training_job_%j.err

# Load the required modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load poetry/1.5.1-GCCcore-12.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
