import contextlib
from typing import Dict, Optional

import deepspeed
import torch
from transformers import AutoConfig
from transformers.integrations.deepspeed import (
    deepspeed_config,
    is_deepspeed_zero3_enabled,
)

from speechlmm.model.decoders.audio_decoder import EncodecDecoder, MimiDecoder
from speechlmm.model.decoders.talking_head import (
    MoshiBertTalkingHead,
    NARTalkingHead,
)

from .text_decoder import LlamaDecoder, MistralDecoder


def build_text_decoder(
    name_or_path: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    add_lm_head: bool = True,
    tokenizer_padding_side: str = "right",
    conversation_version: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
):
    config_dict = config_dict or {}
    name_or_path = name_or_path or config_dict.pop("_name_or_path", None)
    if name_or_path is None:
        raise ValueError(
            "`name_or_path` must be provided either as an explicit argument "
            'or indirectly via `config_dict["_name_or_path"]`.'
        )

    common_kwargs = {
        "name_or_path": name_or_path,
        "config_dict": config_dict,
        "add_lm_head": add_lm_head,
        "tokenizer_padding_side": tokenizer_padding_side,
        "conversation_version": conversation_version,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
    }
    # TODO(anferico): replace if-else with registry pattern
    # e.g. return get_text_decoder(model_type, **common_kwargs)
    model_type = (
        config_dict.get("model_type")
        or AutoConfig.from_pretrained(name_or_path).model_type
    )
    if model_type in ["vicuna", "llama"]:
        return LlamaDecoder(**common_kwargs)
    elif model_type == "mistral":
        return MistralDecoder(**common_kwargs)
    else:
        raise ValueError(f"Unknown text decoder '{model_type}'.")


def build_codec_decoder(
    name_or_path: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
):
    config_dict = config_dict or {}
    name_or_path = name_or_path or config_dict.pop("_name_or_path", None)
    if name_or_path is None:
        raise ValueError(
            "`name_or_path` must be provided either as an explicit argument "
            'or indirectly via `config_dict["_name_or_path"]`.'
        )

    common_kwargs = {
        "name_or_path": name_or_path,
        "config_dict": config_dict,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
    }
    # TODO(anferico): replace if-else with registry pattern
    # e.g. return get_text_decoder(model_type, **common_kwargs)
    model_type = (
        config_dict.get("model_type")
        or AutoConfig.from_pretrained(name_or_path).model_type
    )
    if model_type == "mimi":
        return MimiDecoder(**common_kwargs)
    if model_type == "encodec":
        return EncodecDecoder(**common_kwargs)

    raise ValueError(f"Unknown audio decoder '{model_type}'.")


def build_talking_head(
    config_dict: Optional[Dict] = None,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
):
    # TODO(st3p99): replace if-else with registry pattern
    config_dict = config_dict or {}
    model_type_from_config = config_dict.pop("model_type", None)
    model_type = model_type_from_config
    config = AutoConfig.for_model(model_type, **config_dict)
    common_kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }

    parameter_partitioning = (
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with parameter_partitioning:  # Partition talking head parameters
        if model_type == "moshi_bert":
            return MoshiBertTalkingHead(**common_kwargs)
        elif model_type == "moshi_qformer":
            raise NotImplementedError(
                "MoshiQformerTalkingHead  is not yet implemented."
            )
        elif model_type == "qformer_talking_head":
            return NARTalkingHead(**common_kwargs)
        elif model_type == "base_bert":
            raise NotImplementedError(
                "BaseBertTalkingHead is not yet implemented."
            )
        elif model_type == "base_qformer":
            raise NotImplementedError(
                "BaseQformerTalkingHead is not yet implemented."
            )
        else:
            raise ValueError(f"Unknown talking head '{model_type}'.")
