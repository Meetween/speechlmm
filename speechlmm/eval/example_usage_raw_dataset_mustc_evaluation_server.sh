mustc_asr=( "en" )
mustc_directions=( "en_fr" "en_it" "en_de" "en_es" )
tokenizer_padding_side="left"
results_dir="$SCRATCH/evaluation_results"
models_paths=( "$PLG_GROUPS_STORAGE/plggmeetween/checkpoints/pretraining-paper-compressor-benchmarking/test1/speechlmm-mistral-pretrain-seamless-bert-encoder-asr-st-maxdur20s" )

for m in "${!models_paths[@]}"; do

    #ASR MUSTC (1)
    bash speechlmm/eval/evaluate_model_sbatch.sh \
        --model_path ${models_paths[m]} \
        --dataset mustc_evaluation_server_en \
        --batch_size 4 \
        --results_dir $results_dir \
        --tokenizer_padding_side $tokenizer_padding_side \
        --data_dir $DATA_HOME


    # ST MUSTC  (4)
    for i in "${!mustc_directions[@]}"; do
        bash speechlmm/eval/evaluate_model_sbatch.sh \
            --model_path ${models_paths[m]} \
            --dataset mustc_evaluation_server_${mustc_directions[i]} \
            --batch_size 4 \
            --results_dir $results_dir \
            --tokenizer_padding_side $tokenizer_padding_side \
            --data_dir $DATA_HOME
    done

# (total jobs = 5)
done
# list dir in $CHECKPOINTS_HOME/pretraining-final-pedavero/test2 and launch ASR on each dir "model_path"
# models_paths=( $(ls -d $CHECKPOINTS_HOME/pretraining-final-pedavero/test2/*) )
