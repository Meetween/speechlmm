bash speechlmm/eval/evaluate_model_sbatch.sh \
    --model_path $CHECKPOINTS_HOME/pretraining-paper-compressor-benchmarking/test1/speechlmm-mistral-pretrain-seamless-bert-encoder-asr-st-maxdur20s \
    --dataset libritts_300_test_clean \
    --data_dir $DATA_HOME \
    --batch_size 10 \
    --now
