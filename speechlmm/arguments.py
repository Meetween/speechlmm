from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import transformers


@dataclass
class DataArguments:
    data_config_path: str = field(
        default=None, metadata={"help": "Path to the training data config."}
    )
    dataloader_debug: bool = field(default=False)
    align_text_to_audio: bool = field(default=True)
    get_timestamps: bool = field(default=False)
    use_text_tokens: bool = field(default=True)
    align_with_whisper: bool = field(default=False)
    restore_punctuation_and_spaces: bool = field(default=True)
    filter_broken_samples: bool = field(default=False)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    audio_input_sampling_rate: Optional[int] = None
    codec_frame_rate: Optional[int] = None
    codec_sampling_rate: Optional[int] = None
    video_input_sampling_rate: Optional[int] = None
    group_dataset_by_task: Dict[str, bool] = field(default_factory=dict)
    organize_eval_dataset_per_task: bool = field(default=False)
    cache_final_datasets: bool = field(default=False)
    rebuild_dataset_cache: bool = field(default=False)
    num_proc_for_preprocessing: int = field(default=1)
    task_weights: Optional[Dict[str, float]] = field(default=None)
    multi_task_sampler: str = field(default="random")
    replacement: bool = field(default=True)
    sampler_target_samples: Optional[Dict[str, int]] = field(default=-1)
    max_condition_audio_duration: Optional[float] = field(default=None)
    variable_batch_size: bool = field(default=False)
    max_length_per_batch: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    modality: str = field(default="image")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4 bits or not."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8 bits or not."},
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    lora_adapters: List[Dict[str, Any]] = field(default_factory=list)
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    num_steps_between_each_restart: Optional[int] = field(default=None)
    lr_min: Optional[float] = field(default=1e-6)
    eval_temperature: Optional[float] = field(default=0)
    eval_max_new_tokens: Optional[int] = field(default=200)
    eval_num_batched_generations: Optional[int] = field(default=2)
    freeze_modules: List[str] = field(default_factory=list)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrained_checkpoint: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
