import os
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig

from .audio_encoder import (
    EncodecEncoder,
    HubertEncoder,
    MimiEncoder,
    SeamlessM4Tv2Encoder,
    UnrestrictedWhisperEncoder,
    Wav2Vec2BertEncoder,
    WhisperEncoder,
)
from .clip_encoder import CLIPVisionTower
from .video_encoder import AutoAvsrEncoder


# TODO(anferico): finish implementing this function
def build_vision_encoder(
    model_type: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    cache_dir: Optional[str] = None,
):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_audio_encoder(
    name_or_path: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    delay_load: bool = False,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
    chunk_size_in_seconds: Optional[float] = None,
    chunk_overlap_in_seconds: float = 0.0,
    chunk_encoding_strategy: str = "loop",  # [batch, loop]
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
        "delay_load": delay_load,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
        "chunk_size_in_seconds": chunk_size_in_seconds,
        "chunk_overlap_in_seconds": chunk_overlap_in_seconds,
        "chunk_encoding_strategy": chunk_encoding_strategy,
    }
    # TODO(anferico): replace if-else with registry pattern
    # e.g. return get_audio_encoder(model_type, **common_kwargs)
    model_type = (
        config_dict.get("model_type")
        or AutoConfig.from_pretrained(name_or_path).model_type
    )
    if model_type == "hubert":
        return HubertEncoder(**common_kwargs)
    if model_type == "wav2vec2-bert":
        return Wav2Vec2BertEncoder(**common_kwargs)
    if model_type == "seamless_m4t_v2":
        return SeamlessM4Tv2Encoder(**common_kwargs)
    if model_type == "whisper":
        return WhisperEncoder(**common_kwargs)
    if model_type == "whisper-unrestricted":
        return UnrestrictedWhisperEncoder(**common_kwargs)

    # TODO(anferico): right now we're not passing these to
    # `CodecEncoder`s, but in the future we might want to
    common_kwargs.pop("chunk_size_in_seconds")
    common_kwargs.pop("chunk_overlap_in_seconds")
    common_kwargs.pop("chunk_encoding_strategy")

    if model_type == "encodec":
        return EncodecEncoder(**common_kwargs)
    if model_type == "mimi":
        return MimiEncoder(**common_kwargs)

    raise ValueError(f"Unknown audio encoder: {model_type}")


def build_video_encoder(
    name_or_path: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    delay_load: bool = False,
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
):
    config_dict = config_dict or {}
    name_or_path = name_or_path or config_dict.pop("_name_or_path", None)

    if name_or_path is None:
        raise ValueError(
            "`name_or_path` must be provided either as an explicit argument "
            'or indirectly via `config_dict["_name_or_path"]`.'
        )

    # NOTE(anferico): for some reason I still can't understand, we are
    # not required to wrap this in a `deepspeed.zero.Init` context
    # contrarily to when we instantiate e.g. the audio adapters and the
    # talking head. In fact, if we do, we get an assertion error from
    # DeepSpeed saying that a parameter is expected to have
    # `ds_status=AVAILABLE`, but it has `ds_status=NOT_AVAILABLE`
    return AutoAvsrEncoder(
        name_or_path=name_or_path,
        config_dict=config_dict,
        delay_load=delay_load,
        torch_dtype=torch_dtype,
        device=device,
    )
