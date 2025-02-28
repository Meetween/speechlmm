#!/bin/bash

# Args:
# --model_path: The name of the model to evaluate.
# --model_name: The name of the folder where results will be saved. If not provided, it will be extracted from the model_path.
# --dataset: The name of the dataset to use.
# --data_dir: Optional. The directory where the datasets are stored. Default is '$DATA_HOME'.
# --results_dir: Optional. The directory to store the results. Default is '$SCRATCH/evaluation_results'.
# --from_yml: Optional. If true, dataset is a path to a yml file with the dataset configuration. data_dir is ignored.
# --batch_size: Optional. The batch size to use for evaluation. Default is 10.
# --now: if true, uses the high priority partition on Athena.
# --tokenizer_padding_side: Optional. The side to pad the input sequences. Default is "left".

if [ ! -d ".git" ]; then
    echo "Error: Must be run from the root directory of the Git repository due to relative paths."
    echo "Example: bash speechlmm/eval/evaluate_model_sbatch.sh --model_path <path> --dataset <dataset_name> [--model_name <model_name>] [--data_dir <data_path>] [--results_dir <results_path>] [--from_yml] [--batch_size <batch_size>] [--now]"
    exit 1
fi
# Default values
data_dir="${DATA_HOME}"
results_dir="${SCRATCH}/evaluation_results"
model_path=""
model_name=""
dataset=""
from_yml="false"
batch_size=10
tokenizer_padding_side="left"
now="false"


exists_config(){
    local model_path=$1
    if [ -f "$model_path/config.json" ]; then
        :
    else
        echo "Error: The model path does not contain a config.json file."
        # print the model path and ask to check if the path is correct
        echo "Model path: $model_path"
        echo "Please check if the model path is correct and contains the config.json file."
        exit 1
    fi
}


usage() {
    echo "Usage: $0 --model_path <path> --dataset <dataset_name> [--model_name <model_name>] [--data_dir <data_path>] [--results_dir <results_path>] [--from_yml] [--batch_size <batch_size>] [--now] [--tokenizer_padding_side <padding_side>]"
    exit 1
}

extract_checkpoint_name_from_path() {
    local path="$1"
    IFS='/' read -ra ADDR <<< "$path"
    local len=${#ADDR[@]}

    # This logic handles the case when the model path is an intermediate checkpoint instead of the final one of the training
    # With this notation, my-model_checkpoint-1000 is the checkpoint at the 1000th step of training,
    # while my-model is the final model of that training run.

    if [[ "${ADDR[$len-1]}" == *"checkpoint"* ]]; then
        echo "${ADDR[$len-2]}_${ADDR[$len-1]}"
    else
        echo "${ADDR[$len-1]}"
    fi
}

# Parse the command-line options
while [ "$1" != "" ]; do
    case $1 in
        --model_path ) shift
                       model_path=$1
                       ;;
        --model_name ) shift
                       model_name=$1
                       ;;
        --dataset    ) shift
                       dataset=$1
                       ;;
        --data_dir   ) shift
                       data_dir=$1
                       ;;
        --results_dir) shift
                       results_dir=$1
                       ;;
        --from_yml   ) from_yml="true"
                       ;;
        --batch_size ) shift
                       batch_size=$1
                       ;;
        --tokenizer_padding_side ) shift
                       tokenizer_padding_side=$1
                       ;;
        --now        ) now="true"
                       ;;
        *            ) usage
                       ;;
    esac
    shift
done


# Check for mandatory options
if [ -z "$model_path" ] || [ -z "$dataset" ]; then
    usage
fi

# Check if the model path contains a config.json file
exists_config $model_path

# if results_dir does not exist, create it
if [ ! -d "$results_dir" ]; then
    mkdir -p $results_dir
fi
# if model_name is not provided, extract it from the model_path
if [ -z "$model_name" ]; then
    model_name=$(extract_checkpoint_name_from_path $model_path)
fi

results_dir="$results_dir/$model_name"
if [ ! -d "$results_dir" ]; then
    mkdir -p $results_dir/logs
fi

# Print the configuration
echo "Configuration:"
echo "model_path: $model_path"
echo "model_name: $model_name"
echo "dataset: $dataset"
echo "data_dir: $data_dir"
echo "results_dir: $results_dir"
echo "from_yml: $from_yml"
echo "batch_size: $batch_size"
echo "tokenizer_padding_side: $tokenizer_padding_side"
echo "now: $now"

if [ "$now" == "true" ]; then
    partition="plgrid-now"
    job_duration="1:00:00"
elif [ "$tokenizer_padding_side" == "left" ] && [ "$batch_size" -gt 1 ]; then
    partition="plgrid-gpu-a100"
    job_duration="1:00:00"
else
    partition="plgrid-gpu-a100"
    job_duration="6:00:00"
fi




sbatch --job-name eval_${model_name}_${dataset} \
    --output=$results_dir/logs/evaluation_%j.out \
    --error=$results_dir/logs/evaluation_%j.err \
    --partition=$partition \
    --time=$job_duration \
    --export=ALL,model_path="$model_path",model_name="$model_name",dataset="$dataset",data_dir="$data_dir",results_dir="$results_dir",from_yml="$from_yml",batch_size="$batch_size",tokenizer_padding_side="$tokenizer_padding_side" \
    speechlmm/eval/submit_evaluation.sbatch
