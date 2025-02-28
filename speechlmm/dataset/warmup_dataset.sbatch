#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=plgmeetween2004-gpu-a100
#SBATCH --partition=plgrid-gpu-a100

module load Miniconda3/23.3.1-0
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate speechlmm

arguments=( --config_path "$config_path" )
if [ -v $rebuild_dataset_cache ]; then
    echo "Rebuilding dataset cache"
    arguments+=( --rebuild_dataset_cache )
fi

srun python speechlmm/dataset/warmup_dataset.py "${arguments[@]}"
