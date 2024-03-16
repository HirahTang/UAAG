#!/bin/bash
#SBATCH --job-name=UAAG_test
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=10-00:05:00
#SBATCH --output=UAAG2.out
python scripts/train_diffusion.py --no_wandb 0 --data_path data/uaag_data_tiny.json