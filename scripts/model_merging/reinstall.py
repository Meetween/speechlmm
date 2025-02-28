import logging
import os
import re
from pathlib import Path

import safetensors.torch

# THIS WILL LOAD THE MULTIMODAL MODELS CONTAINS LLAMA (text_decoder)
mllm_model_path = f"{os.getenv('CHECKPOINTS_HOME')}/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_full_ft/checkpoint-12500"
logging.info(
    f"Loading multimodal model from multiple shards in {mllm_model_path}"
)
mllm_state_dict = {}
import re

# match all .safetensors files
pattern = re.compile(r".*\.safetensors$")
for file in os.listdir(mllm_model_path):
    if pattern.match(file):
        shard_state_dict = safetensors.torch.load_file(
            Path(mllm_model_path, file)
        )
        mllm_state_dict.update(shard_state_dict)
# extract all the keys that start with "text_decoder.model.*"
text_decoder_keys = [
    key.replace("text_decoder.model.", "")
    for key in mllm_state_dict.keys()
    if key.startswith("text_decoder.model.")
]

text_decoder_state_dict = {
    key: mllm_state_dict[f"text_decoder.model.{key}"]
    for key in text_decoder_keys
}

llama_merged_path = f"{os.getenv('CHECKPOINTS_HOME')}/llama-ft-merged"
logging.info(
    f"Loading llama merged model from multiple shards in {llama_merged_path}"
)
llama_state_dict = {}
for file in os.listdir(llama_merged_path):
    if pattern.match(file):
        shard_state_dict = safetensors.torch.load_file(
            Path(llama_merged_path, file)
        )
        llama_state_dict.update(shard_state_dict)
print(
    f"llama_state_dict[model.embed_tokens.weight]: {llama_state_dict['model.embed_tokens.weight'].shape}"
)
# substitute the text_decoder values with the ones from llama
text_decoder_state_dict = {
    f"text_decoder.model.{key}": llama_state_dict[key]
    for key in text_decoder_keys
}

print(
    f"Reinstalling the text_decoder values from llama into the multimodal model"
)
# update the multimodal model with the new text_decoder values
mllm_state_dict.update(text_decoder_state_dict)

print(
    f"mllm_state_dict[text_decoder.model.model.embed_tokens.weight]: {mllm_state_dict['text_decoder.model.model.embed_tokens.weight'].shape}"
)
# save the new multimodal model
from speechlmm.model.modeling_speechlmm import SpeechLmmModel

# load the model from the state_dict
new_path = (
    f"{os.getenv('CHECKPOINTS_HOME')}merged/speechlmm-llama-ft-moshi-merged"
)
config_path = f"{os.getenv('CHECKPOINTS_HOME')}/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_full_ft/checkpoint-12500/config.json"
# load json config and extract text_decoder config
import json

with open(config_path, "r") as f:
    config = json.load(f)

from speechlmm.model.configuration_speechlmm import SpeechLmmConfig

# config["text_decoder"]["vocab_size"] =  config["talking_head"]["text_vocab_size"]
pretrained_config = SpeechLmmConfig.from_dict(config)
breakpoint()
model = SpeechLmmModel(pretrained_config)
breakpoint()
model.load_state_dict(
    mllm_state_dict, strict=False
)  # codec_encoder/decoder is missing
breakpoint()
model.save_pretrained(new_path)
