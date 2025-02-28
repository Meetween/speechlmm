#!/bin/bash

# Define the root directory where the audio directories are located
ROOT_DIR="$SCRATCH/ST2ST-testset/audios"

# Define the model path, adjust this according to your system

# if $1 is not empty, use it as the model path
if [ -z "$1" ]
then
  echo "Using default model path"
  MODEL_PATH="$CHECKPOINTS_HOME/moshi/speechlmm-pretrain-audio-seamless-qformer-llama_3_1-moshi_bert-qformer-features-speech2speech/moshi_bert_s2st/checkpoint-12000"
else
  echo "Using model path: $1"
  MODEL_PATH="$1"
fi

# Define the configuration options
TASK="s2st_cloning"
DEVICE="cuda"
CONV_MODE="llama_3_1"  # Set this to your desired conversation mode
TEMPERATURE=0.5
MAX_NEW_TOKENS=512
LOAD_IN_8BIT=false
LOAD_IN_4BIT=false
CAPTION_ONLY=false
TTS_MAX_LEN=30

# Define the source languages
SOURCE_LANGUAGES=("de" "de" "es" "es" "fr" "fr" "it" "it")
TARGET_LANGUAGE="en"

# Array of directory names
DIRS=("riva-de-de-f-margit-s-001" "riva-de-de-m-sven-s-001" "riva-es-es-f-marina-s-001" "riva-es-es-m-diego-s-001"
      "riva-fr-fr-f-camille-s-001" "riva-fr-fr-m-sebastien-s-001" "riva-it-it-f-federica-s-001" "riva-it-it-m-giovanni-s-001")

# Iterate over each directory and launch the S2ST inference
for i in "${!DIRS[@]}"; do
  DIR="${DIRS[i]}"
  SOURCE_LANGUAGE="${SOURCE_LANGUAGES[i]}"
  AUDIO_PATH="${ROOT_DIR}/${DIR}"

  echo "Running S2ST for directory: ${DIR}, Source Language: ${SOURCE_LANGUAGE}, Target Language: ${TARGET_LANGUAGE}"

  python $SPEECHLMM_ROOT/scripts/inference.py \
    --model-path "${MODEL_PATH}" \
    --task "${TASK}" \
    --device "${DEVICE}" \
    --conv-mode "${CONV_MODE}" \
    --temperature "${TEMPERATURE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --tts-max-len "${TTS_MAX_LEN}" \
    --audios-file-path "${AUDIO_PATH}" \
    --source-language "${SOURCE_LANGUAGE}" \
    --target-language "${TARGET_LANGUAGE}"

done
