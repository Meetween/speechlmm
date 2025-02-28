from pathlib import Path
from typing import Dict, Optional, Union

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from speechlmm.model.configuration_speechlmm import SpeechLmmConfig
from speechlmm.model.modeling_speechlmm import SpeechLmmModel


def load_pretrained_model(
    model_name_or_path: Union[str, Path],
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[str, Dict[str, int]] = "cuda",
    **kwargs,
):
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    config = SpeechLmmConfig.from_pretrained(model_name_or_path, **kwargs)

    torch.set_default_device(device)

    model_name_or_path = Path(model_name_or_path)
    lora_adapters_config_paths = list(
        model_name_or_path.rglob("adapter_config.json")
    )
    if len(lora_adapters_config_paths) == 0:
        model = SpeechLmmModel.from_pretrained(
            model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    else:
        model = SpeechLmmModel(
            config=config, attn_implementation=attn_implementation, **kwargs
        )
        first_lora_adapter_config_path, *other_lora_adapter_config_paths = (
            lora_adapters_config_paths
        )
        # load the first LoRA adapter separately
        first_lora_adapter_path = first_lora_adapter_config_path.parent
        model = PeftModel.from_pretrained(
            model,
            first_lora_adapter_path,
            adapter_name=first_lora_adapter_path.name,
        )

        # load the rest of the LoRA adapters
        for lora_adapter_config_path in other_lora_adapter_config_paths:
            lora_adapter_path = lora_adapter_config_path.parent
            model.load_adapter(
                lora_adapter_path,
                adapter_name=lora_adapter_path.name,
                torch_device=device,
            )

        model = model.merge_and_unload(
            progressbar=True,
            adapter_names=[
                lora_adapter_name for lora_adapter_name in model.peft_config
            ],
        )

    model.requires_grad_(False)
    model.to(dtype=config.torch_dtype)
    model.eval()
    return model
