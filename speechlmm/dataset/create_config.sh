#!/bin/bash

# Ask for user input
read -p "Enter the name of the config file (without extension): " config_name
# Check if config_name is not empty
if [ -z "$config_name" ]; then
    echo "Config name cannot be empty."
    exit 1
fi
read -p "Enter OUTPUTS_TEXT_LIST (relative path to OUTPUTS_TEXT_LIST json - must be in speechlmm root dir e.g. speechlmm/dataset/OUTPUTS_TEXT_LIST.json): " OUTPUTS_TEXT_LIST
read -p "Enter INPUTS_TEXT_LIST (relative path to INPUTS_TEXT_LIST json - must be in speechlmm root dir e.g. speechlmm/dataset/INPUTS_TEXT_LIST.json): " INPUTS_TEXT_LIST

# If OUTPUTS_TEXT_LIST or INPUTS_TEXT_LIST is empty, set it to null
[ -z "$OUTPUTS_TEXT_LIST" ] && OUTPUTS_TEXT_LIST='null'
[ -z "$INPUTS_TEXT_LIST" ] && INPUTS_TEXT_LIST='null'

# Create the config file
echo "# conversation template" > "${config_name}.yml"
echo "OUTPUTS_TEXT_LIST: $OUTPUTS_TEXT_LIST" >> "${config_name}.yml"
echo "INPUTS_TEXT_LIST: $INPUTS_TEXT_LIST" >> "${config_name}.yml"
echo "" >> "${config_name}.yml"
echo "# data" >> "${config_name}.yml"
echo "DATA:" >> "${config_name}.yml"

# Ask for dataset names and their inputs
dataset_count=0
while true; do
    read -p "Enter dataset name (or leave empty to finish): " dataset_name
    if [ -z "$dataset_name" ]; then
        if [ $dataset_count -eq 0 ]; then
            echo "At least one dataset must be provided."
            continue
        else
            break
        fi
    fi
    dataset_count=$((dataset_count+1))

    read -p "Enter datapath for $dataset_name: " datapath
    read -p "Enter task for $dataset_name: " task
    # Check if task is one of 'ASR', 'ST', 'SLU', 'SQA', 'DI', 'TR'
    if [[ "$task" != "ASR" && "$task" != "ST" && "$task" != "SLU" && "$task" != "SQA" && "$task" != "DI" && "$task" != "TR" ]]; then
        echo "Invalid task. Must be one of 'ASR', 'ST', 'SLU', 'SQA', 'DI', 'TR'."
        exit 1
    fi
    read -p "Enter languages for $dataset_name (comma-separated, no spaces): " languages
    IFS=',' read -ra LANG_ARRAY <<< "$languages"

    echo "    - $dataset_name:" >> "${config_name}.yml"
    echo "        datapath: \"$datapath\"" >> "${config_name}.yml"
    echo "        task: \"$task\"" >> "${config_name}.yml"
    echo "        languages: [${LANG_ARRAY[@]}]" >> "${config_name}.yml"
    echo "        partitions:" >> "${config_name}.yml"

    # Ask for partition names and their inputs
    partition_count=0
    while true; do
        read -p "Enter partition name for $dataset_name (or leave empty to finish): " partition
        if [ -z "$partition" ]; then
            if [ $partition_count -eq 0 ]; then
                echo "At least one partition must be provided for each dataset."
                continue
            else
                break
            fi
        fi
        partition_count=$((partition_count+1))

        read -p "Enter amount for $partition (options: ':X%' or 'X1:X2%', leave empty for default ':100%'): " amount
        read -p "Enter min_duration for $partition (leave empty for null): " min_duration
        read -p "Enter max_duration for $partition (leave empty for null): " max_duration
        read -p "Enter destination for $partition (options: 'train', 'eval', 'test'): " destination
        # Check if destination is one of 'train', 'test', 'eval'
        if [[ "$destination" != "train" && "$destination" != "test" && "$destination" != "eval" ]]; then
            echo "Invalid destination. Must be one of 'train', 'test', 'eval'."
            exit 1
        fi

        # If amount is empty, set it to :100%
        [ -z "$amount" ] && amount=:100%
        # If min_duration or max_duration is empty, set it to null
        [ -z "$min_duration" ] && min_duration='null'
        [ -z "$max_duration" ] && max_duration='null'

        echo "            $partition:" >> "${config_name}.yml"
        echo "                amount: \"$amount\"" >> "${config_name}.yml"
        echo "                min_duration: $min_duration" >> "${config_name}.yml"
        echo "                max_duration: $max_duration" >> "${config_name}.yml"
        echo "                destination: \"$destination\"" >> "${config_name}.yml"
    done
done

echo "Config file created as ${config_name}.yml"
