#!/bin/bash
#SBATCH --job-name=gen_pocket
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --array=3
#SBATCH --partition=gpu
#SBATCH --time=5-00:05:00

OUTPUT_LOG=/home/qcx679/hantang/UAAG/slurm_config/generate_pocket.log

CONFIG=/home/qcx679/hantang/UAAG/slurm_config/gen_protein_full.txt

method=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
gen_number=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
gen_start=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})

echo "python /home/qcx679/hantang/UAAG/scripts/construct_pocket_data.py --method ${method} --gen_number ${gen_number} --gen_start ${gen_start}" >> ${OUTPUT_LOG}
python /home/qcx679/hantang/UAAG/scripts/construct_pocket_data.py --method ${method} --gen_number ${gen_number} --gen_start ${gen_start} >> ${OUTPUT_LOG}