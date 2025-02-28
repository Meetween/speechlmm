#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --account=plgmeetween2004-gpu-a100
#SBATCH --gres=gpu:1

module load Miniconda3/23.3.1-0
module load CUDA/12.1.1

eval "$(conda shell.bash hook)"
conda activate speechlmm

if [ "$from_yml" = true ]; then
    srun python speechlmm/eval/eval.py \
                --model-path "$model_path" \
                --dataset "$dataset" \
                --data-dir "$data_dir" \
                --results-dir "$results_dir" \
                --from-yml \
                --batch-size "$batch_size" \
                --tokenizer-padding-side "$tokenizer_padding_side"
else
    srun python speechlmm/eval/eval.py \
                --model-path "$model_path" \
                --dataset "$dataset" \
                --data-dir "$data_dir" \
                --results-dir "$results_dir" \
                --batch-size "$batch_size" \
                --tokenizer-padding-side "$tokenizer_padding_side"
fi
