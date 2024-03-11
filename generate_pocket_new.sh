#!/bin/bash
#SBATCH --job-name=gen_pocket
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=10-00:05:00
OUTPUT_LOG=/home/qcx679/hantang/UAAG/slurm_config/generate_pocket_new.log
echo "python /home/qcx679/hantang/UAAG/scripts/construct_pocket_data.py --protein_dir data/pdb_processed" >> ${OUTPUT_LOG}
python /home/qcx679/hantang/UAAG/scripts/construct_pocket_data.py --protein_dir data/pdb_processed >> ${OUTPUT_LOG}