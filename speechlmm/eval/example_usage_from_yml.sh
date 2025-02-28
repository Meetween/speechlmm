
sources=( "en" "it" "de" "es" "fr" )
directions=( "it_en" "de_en" "es_en" "fr_en" "en_de" )
mustc_directions=( "en_fr" "en_it" "en_de" "en_es" )
tokenizer_padding_side="left"
results_dir="$SCRATCH/evaluation_results"
models_paths=( "$PLG_GROUPS_STORAGE/plggmeetween/checkpoints/pretraining-paper-compressor-benchmarking/test1/speechlmm-mistral-pretrain-seamless-bert-encoder-asr-st-maxdur20s")

for m in "${!models_paths[@]}"; do
    # ASR Covost (5)
    for i in "${!sources[@]}"; do
        bash speechlmm/eval/evaluate_model_sbatch.sh \
            --model_path ${models_paths[m]} \
            --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/covost_${sources[i]}_100_percent.yml \
            --batch_size 16 \
            --results_dir $results_dir \
            --tokenizer_padding_side $tokenizer_padding_side \
            --from_yml
    done

    #ASR MUSTC (test-common) (1)
    bash speechlmm/eval/evaluate_model_sbatch.sh \
        --model_path ${models_paths[m]} \
        --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/mustc_en_100_percent.yml \
        --batch_size 4 \
        --results_dir $results_dir \
        --tokenizer_padding_side $tokenizer_padding_side \
        --from_yml

    # ASR MUSTC (test-he) (1)
    bash speechlmm/eval/evaluate_model_sbatch.sh \
        --model_path ${models_paths[m]} \
        --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/mustc-test-he_en_100_percent.yml \
        --batch_size 4 \
        --results_dir $results_dir \
        --tokenizer_padding_side $tokenizer_padding_side \
        --from_yml


    # ST Covost (5)
    for i in "${!directions[@]}"; do
        bash speechlmm/eval/evaluate_model_sbatch.sh \
            --model_path ${models_paths[m]} \
            --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/covost_${directions[i]}_100_percent.yml \
            --batch_size 16 \
            --results_dir $results_dir \
            --tokenizer_padding_side $tokenizer_padding_side \
            --from_yml
    done

    # ST MUSTC (test-common) (4)
    for i in "${!mustc_directions[@]}"; do
        bash speechlmm/eval/evaluate_model_sbatch.sh \
            --model_path ${models_paths[m]} \
            --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/mustc_${mustc_directions[i]}_100_percent.yml \
            --batch_size 4 \
            --results_dir $results_dir \
            --tokenizer_padding_side $tokenizer_padding_side \
            --from_yml
    done

    # ST MUSTC (test-he) (4)
    for i in "${!mustc_directions[@]}"; do
        bash speechlmm/eval/evaluate_model_sbatch.sh \
            --model_path ${models_paths[m]} \
            --dataset $SPEECHLMM_ROOT/speechlmm/eval/data_configs/mustc-test-he_${mustc_directions[i]}_100_percent.yml \
            --batch_size 4 \
            --results_dir $results_dir \
            --tokenizer_padding_side $tokenizer_padding_side \
            --from_yml
    done

# (total jobs = 20)
done
# list dir in $CHECKPOINTS_HOME/pretraining-final-pedavero/test2 and launch ASR on each dir "model_path"
# models_paths=( $(ls -d $CHECKPOINTS_HOME/pretraining-final-pedavero/test2/*) )
