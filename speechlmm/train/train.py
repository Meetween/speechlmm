# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import logging
import os
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.multiprocessing
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from peft.utils.constants import DUMMY_TARGET_MODULES
from transformers import BitsAndBytesConfig
from transformers.trainer import _is_peft_model

from speechlmm.arguments import DataArguments, TrainingArguments
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.dataset.speechlmm_dataset import SpeechLmmDataset
from speechlmm.dataset.utils import DataCollatorForSupervisedDataset
from speechlmm.eval.metrics import (
    compute_asr_metrics,
    compute_st_metrics,
    compute_vsr_metrics,
)
from speechlmm.model.adapters.utils import (
    count_parameters,
    count_trainable_parameters,
)
from speechlmm.model.configuration_speechlmm import SpeechLmmConfig
from speechlmm.model.modeling_speechlmm import SpeechLmmModel
from speechlmm.model.utils import get_candidate_modules_to_save_for_lora
from speechlmm.train.speechlmm_trainer import SpeechLmmTrainer
from speechlmm.utils import validate_lora_adapter_config
from speechlmm.wandb_utils import get_run_tags

# torch.multiprocessing.set_start_method("spawn", force=True)
# multiprocessing.set_start_method("spawn", force=True)

logger = logging.getLogger(__name__)


def find_linear_modules(model: torch.nn.Module, prefix: str = "") -> List[str]:
    return list(
        {
            name
            for name, module in model.named_modules(prefix=prefix)
            if isinstance(module, torch.nn.Linear)
            and not name.endswith(".lm_head")  # needed for 16-bit
        }
    )
    # TODO(anferico): understand what the comment "needed for 16-bit"
    # above means (it was in the original code) and why we're excluding
    # the lm_head


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset, eval_dataset = None, None
    all_datasets = DatasetsWrapper(data_args)
    if all_datasets.train_dataset is not None:
        if len(all_datasets.train_dataset.keys()) > 1:
            train_dataset = {
                task_name: SpeechLmmDataset(
                    dataset=ds, tokenizer=tokenizer, data_args=data_args
                )
                for task_name, ds in all_datasets.train_dataset.items()
            }
        else:
            train_dataset = SpeechLmmDataset(
                dataset=all_datasets.train_dataset["all_tasks"],
                tokenizer=tokenizer,
                data_args=data_args,
            )
    if all_datasets.eval_dataset is not None:
        eval_dataset = {
            task_name: SpeechLmmDataset(
                dataset=ds, tokenizer=tokenizer, data_args=data_args
            )
            for task_name, ds in all_datasets.eval_dataset.items()
        }
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train(path_to_yaml_config: Union[str, Path]):
    with open(path_to_yaml_config, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    data_args = DataArguments(**config_dict["data"])
    training_args = TrainingArguments(**config_dict["training"])
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            raise ValueError(
                "`hub_model_id` must be provided when `push_to_hub` is True."
            )
        hub_token = training_args.hub_token or os.getenv("HF_TOKEN", None)
        if hub_token is None:
            raise ValueError(
                "Hugging Face token must be provided either explicitly (via"
                "the `hub_token` argument) or via the `HF_TOKEN` environment "
                "variable when `push_to_hub` is True."
            )

    if data_args.multi_task_sampler == "alternating":
        training_args.accelerator_config.split_batches = True

    # Wire task weights and sampler type from data_args to training_args
    if data_args.task_weights is not None:
        training_args.task_weights = data_args.task_weights
    if data_args.sampler_target_samples is not None:
        training_args.sampler_target_samples = data_args.sampler_target_samples
    if data_args.multi_task_sampler is not None:
        training_args.multi_task_sampler = data_args.multi_task_sampler
    if data_args.replacement is not None:
        training_args.replacement = data_args.replacement
    if data_args.variable_batch_size is not None:
        training_args.variable_batch_size = data_args.variable_batch_size
    if data_args.max_length_per_batch is not None:
        training_args.max_length_per_batch = data_args.max_length_per_batch

    if training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    quantization_kwargs = dict()
    if training_args.load_in_4bit or training_args.load_in_8bit:
        quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_4bit,
            load_in_8bit=training_args.load_in_8bit,
            llm_int8_skip_modules=["vision_encoder", "audio_encoder"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=training_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=training_args.bnb_4bit_quant_type,  # fp4, nf4
        )

    model_config = SpeechLmmConfig(
        torch_dtype=torch_dtype, **config_dict["model"], **quantization_kwargs
    )
    model_config.align_text_to_audio = data_args.align_text_to_audio

    if training_args.pretrained_checkpoint is not None:
        model = SpeechLmmModel.from_pretrained(
            training_args.pretrained_checkpoint,
            config=model_config,
            cache_dir=training_args.cache_dir,
            ignore_mismatched_sizes=False,
            # ↑ NOTE(anferico): default is False anyway, but we set it
            # explicitly just to make everyone aware of the existence of
            # this argument. There are cases where we may want to set it
            # to True, e.g. when instantiating a model with 10 labels
            # from a checkpoint that was trained with 5.
            # Note that if you get an error about mismatched sizes,
            # chances are that you either forgot to set
            # `model.add_all_multimodal_tokens=True` when running the
            # first training, or that you are using a checkpoint that
            # was trained with different hyperparameters altogether, in
            # which case setting this to True would be a mistake
            attn_implementation=training_args.attn_implementation,
            freeze_modules=training_args.freeze_modules,
            delay_load_video_encoder=True,
        )
        if model.video_encoder is not None:
            # NOTE(anferico): for some reason we must clarify at some
            # point, instantiating the `Encoder` class (from the
            # Auto-AVSR implementation) inside
            # `AutoAvsrEncoder.__init__` is problematic when we use
            # DeepSpeed ZeRO-3 in a training in which we resume from a
            # pre-trained checkpoint. More specifically, we get an
            # `AssertionError` when running `AutoAvsrEncoder.forward`
            # when running `AutoAvsrEncoder.forward` (a parameter is
            # expected to have `ds_status=AVAILABLE`, but it has
            # `ds_status=NOT_AVAILABLE` instead.
            # My suspect is that since `SpeechLmmModel.from_pretrained`
            # instantiates `SpeechLmmModel` inside a
            # `deepspeed.zero.Init` context manager, that breaks things
            # because apparently the parameters of `Encoder` have
            # already been partitioned via another `deepspeed.zero.Init`
            # (not sure where), so doing it twice results in the
            # assertion error
            model.video_encoder._load_model()
    else:
        model = SpeechLmmModel(
            model_config,
            attn_implementation=training_args.attn_implementation,
            cache_dir=training_args.cache_dir,
            freeze_modules=training_args.freeze_modules,
        )

    data_args.conversation_version = (
        model.config.text_decoder.conversation_version
    )

    # TODO(anferico): figure out exactly why this is needed. If not set
    # to False, the LLM's forward() method (at least Mistral's) will
    # error out while computing the evaluation loss saying that it is
    # expecting left-padded inputs when using Flash Attention 2.
    # Question:
    # - Is it safe to just set the tokenizer padding side to left before
    #   tokenizing the batches used to compute the eval loss? Right now,
    #   we only do that for the batches used to perform generation while
    #   training
    model.text_decoder.config.use_cache = (
        training_args.attn_implementation != "flash_attention_2"
    )

    # TODO(anferico): check whether it's safe to merge this "if" with
    # the one further below (it has the exact same condition)
    if "quantization_config" in quantization_kwargs:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    if training_args.gradient_checkpointing:
        # TODO(anferico): why do we need to do this?
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    if len(training_args.lora_adapters) > 0:
        # collect non-LoRA trainables *before* adding any LoRA adapter
        modules_to_save = [
            module_name
            for module_name, module in get_candidate_modules_to_save_for_lora(
                model
            )
            if any(p.requires_grad for p in module.parameters())
        ]

        logger.info("Adding LoRA adapters...")
        all_adapter_names = []
        for adapter_config in training_args.lora_adapters:
            validate_lora_adapter_config(adapter_config)

            get_target_module = attrgetter(adapter_config["target_module"])
            if _is_peft_model(model):
                target_module = get_target_module(model.get_base_model())
            else:
                target_module = get_target_module(model)

            lora_config = LoraConfig(
                task_type=adapter_config["task_type"],
                r=adapter_config["r"],
                target_modules=find_linear_modules(
                    target_module, prefix=adapter_config["target_module"]
                ),
                lora_alpha=adapter_config["lora_alpha"],
                lora_dropout=adapter_config["lora_dropout"],
                bias=adapter_config["bias"],
                use_rslora=adapter_config["use_rslora"],
            )

            adapter_name = adapter_config["name"]
            if not _is_peft_model(model):  # 1st LoRA being added
                model = get_peft_model(
                    model,
                    peft_config=lora_config,
                    adapter_name=adapter_name,
                )
            else:  # (n+1)th LoRA being added
                model.add_adapter(
                    adapter_name=adapter_name,
                    peft_config=lora_config,
                )

            all_adapter_names.append(adapter_name)

        if len(modules_to_save) > 0:
            dummy_adapter_name = "other_modules"
            dummy_adapter_config = LoraConfig(
                target_modules=DUMMY_TARGET_MODULES,
                modules_to_save=modules_to_save,
            )
            model.add_adapter(
                adapter_name=dummy_adapter_name,
                peft_config=dummy_adapter_config,
            )

            all_adapter_names.append(dummy_adapter_name)

        # set the added adapters as trainable
        model.base_model.set_adapter(all_adapter_names)
    if "quantization_config" in quantization_kwargs:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch_dtype)
            if "norm" in name:
                # cast LayerNorm to FP32 for stability
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(torch_dtype)

    logger.info(f"Total parameters: {count_parameters(model)}")
    logger.info(f"Trainable parameters: {count_trainable_parameters(model)}")
    if model.audio_encoder is not None:
        data_args.audio_input_sampling_rate = (
            model.audio_encoder.input_sampling_rate
        )
    if model.codec_encoder is not None:
        data_args.codec_output_sampling_rate = (
            model.codec_decoder.output_sampling_rate
        )
        data_args.codec_input_sampling_rate = (
            model.codec_decoder.input_sampling_rate
        )

    data_module = make_supervised_data_module(
        tokenizer=model.text_decoder.tokenizer, data_args=data_args
    )
    compute_metrics_on_generate_per_task = (
        {
            "ASR": compute_asr_metrics,
            "ST": compute_st_metrics,
            "VSR": compute_vsr_metrics,
        }
        if training_args.evaluation_strategy != "no"
        else None
    )
    logging.info(f"Start training")
    trainer = SpeechLmmTrainer(
        compute_metrics_per_task=None,
        compute_metrics_on_generate_per_task=compute_metrics_on_generate_per_task,
        model=model,
        tokenizer=model.text_decoder.tokenizer,
        args=training_args,
        wandb_tags=get_run_tags(model.config, data_args, training_args),
        save_trainable_params_only=True,
        **data_module,
    )
    # TODO(anferico): dirty hack to let the model access the current
    # training step (needed by talking_head). Ideally we should find a
    # better way to do this
    if _is_peft_model(model):
        model.get_base_model().trainer = trainer
    else:
        model.trainer = trainer

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    model.config.use_cache = True

    # also triggers a push to the hub if push_to_hub is True
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    train(args.config)
