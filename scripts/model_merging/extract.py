import logging
import os
import re
from pathlib import Path

import safetensors.torch

pretrained_model_path = f"{os.getenv('CHECKPOINTS_HOME')}/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_full_ft/checkpoint-12500"

logging.info(f"Loading model from multiple shards in {pretrained_model_path}")
state_dict = {}
import re

# match all .safetensors files
pattern = re.compile(r".*\.safetensors$")
for file in os.listdir(pretrained_model_path):
    if pattern.match(file):
        shard_state_dict = safetensors.torch.load_file(
            Path(pretrained_model_path, file)
        )
        state_dict.update(shard_state_dict)

# extract all the keys that start with "text_decoder.model.*"
text_decoder_keys = [
    key.replace("text_decoder.model.", "")
    for key in state_dict.keys()
    if key.startswith("text_decoder.model.")
]

state_dict = {
    key: state_dict[f"text_decoder.model.{key}"] for key in text_decoder_keys
}

# import LlamaForCausalLM
from transformers import LlamaForCausalLM

# load the model from the state_dict
new_path = f"{os.getenv('SCRATCH')}/merged/llama-ft-to-merge"
config_path = f"{os.getenv('CHECKPOINTS_HOME')}/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_full_ft/checkpoint-12500/config.json"
# load json config and extract text_decoder config
import json

with open(config_path, "r") as f:
    config = json.load(f)

text_decoder_config = config["text_decoder"]

from transformers import PretrainedConfig

pretrained_config = PretrainedConfig.from_dict(text_decoder_config)
pretrained_config.vocab_size = config["talking_head"]["text_vocab_size"]
model = LlamaForCausalLM(pretrained_config)
model.load_state_dict(state_dict)
model.save_pretrained(new_path)

# TODO: copy also the tokenizer .json
