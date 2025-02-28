import logging
import re
from typing import List, Optional, Tuple, Union
from unittest import mock

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_utils import set_initialized_submodules

from speechlmm.constants import (
    DEFAULT_AUDIO_CONDITION_END_TOKEN,
    DEFAULT_AUDIO_CONDITION_START_TOKEN,
    DEFAULT_AUDIO_EPAD_TOKEN,
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
    DEFAULT_AUDIO_OUTPUT_END_TOKEN,
    DEFAULT_AUDIO_OUTPUT_START_TOKEN,
    DEFAULT_AUDIO_PAD_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_VIDEO_INPUT_END_TOKEN,
    DEFAULT_VIDEO_INPUT_START_TOKEN,
    IGNORE_INDEX,
    LORA_ADAPTERS_DIR,
    VIDEO_TOKEN_INDEX,
)
from speechlmm.model.adapters.builder import (
    BackfeedingAdapterOnFeatures,
    CifAdapter,
    CtcAdapter,
    MlpAdapter,
    WindowLevelQformerAdapter,
    build_audio_adapter,
    build_backfeeding_adapter,
    build_video_adapter,
    build_vision_adapter,
)
from speechlmm.model.adapters.outputs import (
    CifAdapterOutput,
    CodecOutput,
    CtcAdapterOutput,
    SpeechLmmModuleOutput,
)
from speechlmm.model.configuration_speechlmm import SpeechLmmConfig
from speechlmm.model.decoders.builder import (
    build_codec_decoder,
    build_talking_head,
    build_text_decoder,
)
from speechlmm.model.decoders.talking_head import (
    MoshiBertTalkingHead,
    NARTalkingHead,
)
from speechlmm.model.encoders.builder import (
    build_audio_encoder,
    build_video_encoder,
    build_vision_encoder,
)
from speechlmm.model.model_outputs import (
    CausalLMOutputWithPastAndGranularLosses,
    CausalLMOutputWithPastAndGranularLossesAndMetrics,
    TalkingHeadOutput,
)
from speechlmm.model.sampling import sample_token
from speechlmm.model.text_audio_aligner import (
    AlignCodesAndText,
    AlignWithUnkownTokens,
)


def set_initialized_submodules_patched(model, state_dict_keys):
    not_initialized_submodules = set_initialized_submodules(
        model, state_dict_keys
    )
    pretrained_modules = ["audio_encoder", "video_encoder", "text_decoder"]
    not_initialized_submodule_names = list(not_initialized_submodules.keys())
    for k in not_initialized_submodule_names:
        if any(k.startswith(m) for m in pretrained_modules):
            not_initialized_submodules.pop(k)
    return not_initialized_submodules

class SpeechLmmPreTrainedModel(PreTrainedModel):
    config_class = SpeechLmmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        hf_quantizer=None,
        keep_in_fp32_modules=None,
        gguf_path=None,
    ):
        # NOTE(anferico): rationale: when using DeepSpeed ZeRO-3,
        # `PreTrainedModel._load_pretrained_model` performs a
        # `deepspeed.zero.GatheredParameters` on the parameters that it
        # can't find in the state dict before calling
        # `_initialize_weights` on the corresponding modules. Combined
        # with the fact that we don't include non-trainable parameters
        # in the state dict, this is problematic because we don't want
        # to waste memory for gathering and initializing parameters that
        # will be overwritten later when `SpeechLmmModel` loads its
        # pretrained modules like `audio_encoder`, `video_encoder` and
        # `text_decoder`.
        #
        # The solution is to patch `set_initialized_submodules` so that
        # the pretrained modules in `SpeechLmmModel` are not included among
        # the submodules that need to be initialized
        with mock.patch(
            "transformers.modeling_utils.set_initialized_submodules",
            new=set_initialized_submodules_patched,
        ):
            return super(SpeechLmmPreTrainedModel, cls)._load_pretrained_model(
                model,
                state_dict,
                loaded_keys,
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
                gguf_path=gguf_path,
            )


class SpeechLmmModel(SpeechLmmPreTrainedModel):
    def __init__(
        self,
        config: SpeechLmmConfig,
        attn_implementation: Optional[str] = None,
        cache_dir: Optional[str] = None,
        freeze_modules: Optional[List[str]] = None,
        delay_load_video_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.text_decoder = build_text_decoder(
            config_dict=self.config.text_decoder.to_dict(),
            add_lm_head=self.config.add_lm_head,
            tokenizer_padding_side=self.config.tokenizer_padding_side,
            conversation_version=self.config.text_decoder.conversation_version,
            attn_implementation=attn_implementation,
            torch_dtype=self.config.torch_dtype,
            cache_dir=cache_dir,
        )

        self.attn_implementation = attn_implementation
        self.cache_dir = cache_dir
        self.freeze_modules = freeze_modules or []

        if getattr(self.config, "add_all_multimodal_tokens", False):
            self._extend_tokenizer_with_audio_input_tokens()
            # self._extend_tokenizer_with_audio_output_tokens()
            self._extend_tokenizer_with_video_input_tokens()
            # skipping vision tokens for now

        self.vision_encoder, self.vision_adapter = None, None
        if hasattr(self.config, "vision_encoder"):
            self._initialize_vision_modules()

        self.audio_encoder, self.audio_adapter = None, None
        if hasattr(self.config, "audio_encoder"):
            self._initialize_audio_input_modules()

        self.codec_encoder = None
        self.codec_decoder = None
        self.backfeeding_audio_adapter = None
        self.conditioning_audio_adapter = None
        self.talking_head = None
        if hasattr(self.config, "codec_decoder"):
            self._initialize_audio_output_modules()
            logging.info(
                f"Using codebook weights: {self.config.codebook_weights}"
            )
            self.codebook_weights = self.config.codebook_weights[
                : self.codec_decoder.num_quantizers
            ]
            # self.codebook_weights = [10] + [1] * (self.codec_decoder.num_quantizers - 1)
            # self.codebook_weights = [1] * self.codec_decoder.num_quantizers

        self.decay_fn = lambda w_0, step_ratio, decay_exp: w_0 * (
            (1 - step_ratio) ** decay_exp
        )

        self.inverse_decay_fn = lambda w_0, step_ratio, decay_exp: 1 - (
            1 - w_0
        ) * ((1 - step_ratio) ** decay_exp)

        self.video_encoder, self.video_adapter = None, None
        if hasattr(self.config, "video_encoder"):
            self._initialize_video_input_modules(delay_load_video_encoder)

        self._set_trainable_parameters()
        self.post_init()

    def _initialize_vision_modules(self):
        self.vision_encoder = build_vision_encoder(
            model_type=self.config.vision_encoder.model_type,
            config_dict=self.config.vision_encoder.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
            cache_dir=self.cache_dir,
        )
        self.vision_adapter = build_vision_adapter(
            config_dict=self.config.vision_adapter.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
        )
        self.vision_adapter.to(dtype=self.config.torch_dtype)

        self._extend_tokenizer_with_image_tokens()

        if "unpad" in self.config.get("vision_patch_merge_type", ""):
            embed_std = 1 / torch.sqrt(
                torch.tensor(
                    self.config.text_decoder.hidden_size, dtype=self.dtype
                )
            )
            self.image_newline = nn.Parameter(
                torch.randn(
                    self.config.text_decoder.hidden_size, dtype=self.dtype
                )
                * embed_std
            )

    def _extend_tokenizer_with_image_tokens(self):
        if self.config.mm_use_im_start_end:
            self.text_decoder._resize_tokenizer_and_embedding_layer(
                additional_special_tokens=[
                    DEFAULT_IM_START_TOKEN,
                    DEFAULT_IM_END_TOKEN,
                ]
            )
        if self.config.vision_use_patch_token:
            self.text_decoder._resize_tokenizer_and_embedding_layer(
                additional_special_tokens=[DEFAULT_IMAGE_PATCH_TOKEN],
            )

    def _initialize_audio_input_modules(self):
        self.audio_encoder = build_audio_encoder(
            config_dict=self.config.audio_encoder.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
            cache_dir=self.cache_dir,
            chunk_size_in_seconds=self.config.chunk_size_in_seconds,
            chunk_overlap_in_seconds=self.config.chunk_overlap_in_seconds,
            chunk_encoding_strategy=self.config.chunk_encoding_strategy,
        )

        self.audio_adapter = build_audio_adapter(
            audio_features_sampling_rate=self.audio_encoder.output_sampling_rate,
            config_dict=self.config.audio_adapter.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
        )
        self._extend_tokenizer_with_audio_input_tokens()

    def _initialize_audio_output_modules(self):
        if self.config.use_audio_encoder_as_codec_encoder:
            self.codec_encoder = self.audio_encoder
        else:
            self.codec_encoder = build_audio_encoder(
                config_dict=self.config.codec_encoder.to_dict(),
                attn_implementation=self.attn_implementation,
                torch_dtype=self.config.torch_dtype,
                cache_dir=self.cache_dir,
            )

        self.codec_decoder = build_codec_decoder(
            config_dict=self.config.codec_decoder.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
            cache_dir=self.cache_dir,
        )
        self.backfeeding_audio_adapter = build_backfeeding_adapter(
            audio_features_sampling_rate=self.codec_encoder.output_sampling_rate,
            config_dict=self.config.backfeeding_audio_adapter.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
        )

        if hasattr(self.config, "conditioning_audio_adapter"):
            self.conditioning_audio_adapter = build_audio_adapter(
                audio_features_sampling_rate=self.codec_encoder.output_sampling_rate,
                config_dict=self.config.conditioning_audio_adapter.to_dict(),
                attn_implementation=self.attn_implementation,
                torch_dtype=self.config.torch_dtype,
            )
            print("conditioning_audio_adapter")

        self._extend_tokenizer_with_audio_output_tokens()
        # ↑ this also updates `vocab_size` in `text_decoder.config`

        self.config.talking_head.text_vocab_size = (
            self.text_decoder.config.vocab_size
        )
        if self.config.talking_head.model_type == "qformer_talking_head":
            self.config.talking_head.depformer_expansion_factor = getattr(
                self.backfeeding_audio_adapter.adapter.config,
                "compress_factor",
                1,
            )
            self.config.talking_head.cross_attention_window_size = getattr(
                self.backfeeding_audio_adapter.adapter.config,
                "num_queries",
                1,
            )

        self.talking_head_use_text_tokens = getattr(
            self.config.talking_head, "use_text_tokens", True
        )
        print(
            f"WARNING: talking_head_use_text_tokens is {self.talking_head_use_text_tokens}"
        )

        self.talking_head = build_talking_head(
            config_dict=self.config.talking_head.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
            cache_dir=self.cache_dir,
        )

    # TODO(anferico): merge this method and `_extend_tokenizer_with_image_tokens`
    def _extend_tokenizer_with_audio_input_tokens(self):
        self.text_decoder._resize_tokenizer_and_embedding_layer(
            additional_special_tokens=[
                DEFAULT_AUDIO_INPUT_START_TOKEN,
                DEFAULT_AUDIO_INPUT_END_TOKEN,
            ],
        )
        is_audio_adapter_trainable = (
            getattr(self, "audio_adapter", None) is not None
            and self.audio_adapter.get_trainable_parameters()["total"] > 0
        )
        if is_audio_adapter_trainable:
            self.text_decoder.get_input_embeddings().requires_grad_(True)
            self.text_decoder.get_output_embeddings().requires_grad_(False)

    def _extend_tokenizer_with_audio_output_tokens(self):
        # Consolidate basic audio tokens
        self.text_decoder._resize_tokenizer_and_embedding_layer(
            additional_special_tokens=[
                DEFAULT_AUDIO_OUTPUT_START_TOKEN,
                DEFAULT_AUDIO_OUTPUT_END_TOKEN,
                DEFAULT_AUDIO_PAD_TOKEN,
            ],
        )

        # Add alignment token if needed
        if self.config.align_text_to_audio:
            self.text_decoder._resize_tokenizer_and_embedding_layer(
                additional_special_tokens=[DEFAULT_AUDIO_EPAD_TOKEN]
            )

        # Add conditioning tokens if audio adapter exists or if configured
        if getattr(
            self, "conditioning_audio_adapter", None
        ) is not None or getattr(
            self.config, "add_all_multimodal_tokens", False
        ):
            self.text_decoder._resize_tokenizer_and_embedding_layer(
                additional_special_tokens=[
                    DEFAULT_AUDIO_CONDITION_START_TOKEN,
                    DEFAULT_AUDIO_CONDITION_END_TOKEN,
                ]
            )

        # FIXME(st3p99): remove from here
        self.use_sampling = False
        self.temp_text = 0.5
        self.top_k_text = 75
        self.temp_audio = 0.8
        self.top_k_audio = 250

    def _initialize_video_input_modules(
        self, delay_load_video_encoder: bool = False
    ):
        self.video_encoder = build_video_encoder(
            config_dict=self.config.video_encoder.to_dict(),
            delay_load=delay_load_video_encoder,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
            device=(
                "cuda"
                if torch.cuda.is_available()
                else torch.get_default_device()
            ),
            cache_dir=self.cache_dir,
        )
        self.video_adapter = build_video_adapter(
            video_features_sampling_rate=self.video_encoder.input_sampling_rate,
            config_dict=self.config.video_adapter.to_dict(),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.config.torch_dtype,
        )
        # This is generally redundant, but just to make sure
        self.video_adapter.to(dtype=self.config.torch_dtype)

        self._extend_tokenizer_with_video_input_tokens()

    # TODO(anferico): merge this method and `_extend_tokenizer_with_image_tokens`
    def _extend_tokenizer_with_video_input_tokens(self):
        self.text_decoder._resize_tokenizer_and_embedding_layer(
            additional_special_tokens=[
                DEFAULT_VIDEO_INPUT_START_TOKEN,
                DEFAULT_VIDEO_INPUT_END_TOKEN,
            ]
        )

    def _set_trainable_parameters(self):
        self.requires_grad_(True)
        for module_name in self.freeze_modules:
            module = getattr(self, module_name)
            if module is not None:
                module.requires_grad_(False)

        is_text_decoder_trainable = "text_decoder" not in self.freeze_modules
        is_vision_adapter_trainable = (
            self.vision_adapter is not None
            and "vision_adapter" not in self.freeze_modules
        )
        is_audio_adapter_trainable = (
            self.audio_adapter is not None
            and "audio_adapter" not in self.freeze_modules
        )
        is_video_adapter_trainable = (
            self.video_adapter is not None
            and "video_adapter" not in self.freeze_modules
        )
        is_backfeeding_audio_adapter_trainable = (
            self.backfeeding_audio_adapter is not None
            and "backfeeding_audio_adapter" not in self.freeze_modules
        )

        self.text_decoder.get_input_embeddings().requires_grad_(
            is_text_decoder_trainable
            or (
                is_vision_adapter_trainable and self.config.mm_use_im_start_end
            )
            or is_audio_adapter_trainable
            or is_video_adapter_trainable
            or is_backfeeding_audio_adapter_trainable
        )
        if not self.text_decoder.config.tie_word_embeddings:
            self.text_decoder.get_output_embeddings().requires_grad_(
                is_text_decoder_trainable
                or is_backfeeding_audio_adapter_trainable
            )

    def get_input_embeddings(self):
        return self.text_decoder.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_audios_srs: Optional[List[Tuple[torch.FloatTensor, int]]] = None,
        condition_audios_srs: Optional[
            List[Tuple[torch.FloatTensor, int]]
        ] = None,
        output_audios_srs: Optional[
            List[Tuple[torch.FloatTensor, int]]
        ] = None,
        transcription_ids: Optional[torch.LongTensor] = None,
        transcription_attention_mask: Optional[torch.Tensor] = None,
        aligned_transcription_ids: Optional[torch.LongTensor] = None,
        input_videos_srs: Optional[List[Tuple[torch.FloatTensor, int]]] = None,
        video_transcription_ids: Optional[torch.LongTensor] = None,
        video_transcription_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndGranularLosses]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                raw_audio_adapter_output,
                raw_video_adapter_output,
            ) = self.prepare_multimodal_inputs_labels(
                text_inputs={
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "labels": labels,
                },
                modalities_args={
                    "audio": {
                        "input_audios_srs": input_audios_srs,
                        "condition_audios_srs": condition_audios_srs,
                        "output_audios_srs": output_audios_srs,
                        "transcription_ids": transcription_ids,
                        "transcription_attention_mask": (
                            transcription_attention_mask
                        ),
                        "aligned_transcription_ids": aligned_transcription_ids,
                    },
                    "image": {
                        "images": images,
                        "image_sizes": image_sizes,
                    },
                    "video": {
                        "input_videos_srs": input_videos_srs,
                        "video_transcription_ids": video_transcription_ids,
                        "video_transcription_attention_mask": (
                            video_transcription_attention_mask
                        ),
                    },
                },
            )

        # Synchronize GPUs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        model_output = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=(
                output_hidden_states if not self.talking_head else True
            ),
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if raw_audio_adapter_output is not None:
            model_output = self.update_model_output_with_audio_adapter_output(
                model_output,
                [
                    raw_audio_adapter_output.get("audio_input", None),
                    raw_audio_adapter_output.get("audio_condition", None),
                ],
            )
            if (
                self.talking_head is not None
                and raw_audio_adapter_output.get("audio_output") is not None
            ):
                codes = raw_audio_adapter_output.get("audio_output").codes
                codes_padding_mask = raw_audio_adapter_output.get(
                    "audio_output"
                ).padding_mask
                durations = getattr(
                    raw_audio_adapter_output.get("audio_output"),
                    "durations",
                    None,
                )
                windowed_cross_attention_mask = getattr(
                    raw_audio_adapter_output.get("audio_output"),
                    "windowed_cross_attention_mask",
                    None,
                )
                model_output = self._talking_head_forward(
                    model_output,
                    attention_mask,
                    codes,
                    codes_padding_mask,
                    labels,
                    durations,
                    windowed_cross_attention_mask,
                )
        elif raw_video_adapter_output is not None:
            model_output = self.update_model_output_with_video_adapter_output(
                model_output, raw_video_adapter_output
            )
        else:
            model_output = self.update_model_output_with_audio_adapter_output(
                model_output, []
            )

        # Synchronize GPUs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return model_output

    def _talking_head_forward(
        self,
        model_output: CausalLMOutputWithPastAndGranularLosses,
        attention_mask: torch.BoolTensor,
        codes: torch.LongTensor,
        codes_padding_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
        durations: Optional[torch.LongTensor] = None,
        windowed_cross_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> CausalLMOutputWithPastAndGranularLosses:

        hidden_states = model_output.hidden_states[-1]

        cross_entropy_weight = torch.ones(
            self.text_decoder.config.vocab_size
        ).to(hidden_states.device)

        cur_step_max_step_ratio = (
            self.trainer.state.global_step / self.trainer.state.max_steps
        )
        if isinstance(self.talking_head, MoshiBertTalkingHead):
            cross_entropy_weight[
                self.text_decoder.tokenizer.get_added_vocab()[
                    DEFAULT_AUDIO_PAD_TOKEN
                ]
            ] = (
                self.decay_fn(
                    self.config.pad_audio_weight,
                    cur_step_max_step_ratio,
                    self.config.pad_epad_audio_weight_decay,
                )
                if self.config.align_text_to_audio
                else self.config.pad_audio_weight
            )

            if self.config.align_text_to_audio:
                cross_entropy_weight[
                    self.text_decoder.tokenizer.get_added_vocab()[
                        DEFAULT_AUDIO_EPAD_TOKEN
                    ]
                ] = self.decay_fn(
                    self.config.epad_audio_weight,
                    cur_step_max_step_ratio,
                    self.config.pad_epad_audio_weight_decay,
                )

            text_loss = None
            logits = self.text_decoder.get_output_embeddings()(hidden_states)
            logits = logits.float()
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(
                    weight=cross_entropy_weight, ignore_index=IGNORE_INDEX
                )
                shift_logits = shift_logits.view(
                    -1, self.text_decoder.config.vocab_size
                )
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                text_loss = loss_fct(shift_logits, shift_labels)

            model_output.loss = text_loss

        def create_padded_batch(tensors, attn_mask):
            lengths = torch.tensor([t.size(0) for t in tensors])
            max_len = lengths.max()
            padded_batch = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True
            )
            if attn_mask is None:
                mask = torch.arange(max_len)[None, :] < lengths[:, None]
            else:
                mask = torch.nn.utils.rnn.pad_sequence(
                    attn_mask, batch_first=True
                )
            return padded_batch, mask

        starts = []
        ends = []
        batch_indexes = []
        start_placeholder = self.text_decoder.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
        ]
        end_placeholder = self.text_decoder.tokenizer.get_added_vocab()[
            DEFAULT_AUDIO_OUTPUT_END_TOKEN
        ]
        for batch_idx, cur_labels in enumerate(labels):
            start = torch.where(cur_labels == start_placeholder)[0].tolist()
            end = torch.where(cur_labels == end_placeholder)[0].tolist()
            assert len(start) == len(end), "Start and end tokens must match"
            if start and end:
                starts.extend(start)
                ends.extend(end)
                batch_indexes.append(batch_idx)

        text_tokens = []
        context_vector = []
        context_vector_attention_mask = []
        for batch_idx, s, e in zip(batch_indexes, starts, ends):
            text_tokens.append(labels[batch_idx, s + 1 : e])
            context_vector.append(hidden_states[batch_idx, s + 1 : e, :])
            context_vector_attention_mask.append(
                attention_mask[batch_idx, s + 1 : e]
            )
        # context_vector = [
        #     h[s + 1 : e, :]
        #     for h, s, e in zip(
        #         hidden_states,
        #         starts,
        #         ends,
        #     )
        # ]
        # context_vector_attention_mask = [
        #     a[s + 1 : e]
        #     for a, s, e in zip(
        #         attention_mask,
        #         starts,
        #         ends,
        #     )
        # ]
        # text_tokens = [
        #     l[s + 1 : e]
        #     for l, s, e in zip(
        #         labels,
        #         starts,
        #         ends,
        #     )
        # ]
        text_tokens = torch.nn.utils.rnn.pad_sequence(
            text_tokens, batch_first=True
        )

        context_vector, context_vector_attention_mask = create_padded_batch(
            context_vector, context_vector_attention_mask
        )

        talking_head_input = {
            "context_vector": context_vector,
            "context_attention_mask": context_vector_attention_mask,
        }

        if isinstance(self.talking_head, MoshiBertTalkingHead):
            talking_head_input["codes"] = codes
        elif isinstance(self.talking_head, NARTalkingHead):
            talking_head_input["durations_gt"] = durations
        if self.config.align_text_to_audio or isinstance(
            self.talking_head, MoshiBertTalkingHead
        ):
            talking_head_input["text_tokens"] = text_tokens

        # Let's talk!
        talking_head_output = self.talking_head(**talking_head_input)

        audio_loss, metrics_per_codebook = self.compute_audio_loss_and_metrics(
            talking_head_output,
            codes,
            codes_padding_mask,
            windowed_cross_attention_mask,
        )

        return self.update_model_output_with_talking_head_output(
            model_output, audio_loss, metrics_per_codebook
        )

    def compute_audio_loss_and_metrics(
        self,
        talking_head_output: TalkingHeadOutput,
        codes: torch.LongTensor,
        codes_padding_mask: torch.BoolTensor,
        windowed_cross_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        # FIXME hardcode for compression in output
        logits = talking_head_output.logits

        new_labels = (
            codes.clone()
            .masked_fill_(
                codes_padding_mask.unsqueeze(-1).expand(
                    -1, -1, self.codec_decoder.num_quantizers
                ),
                IGNORE_INDEX,
            )
            .to(logits.device)
        )

        # get the valid indices
        valid_indices = new_labels > 0

        logits = logits.float()
        logits = logits[:, :, :, :].contiguous()
        if (
            self.config.talking_head.model_type == "qformer_talking_head"
            and self.talking_head.depformer_expansion_factor > 1
        ):
            max_len = (
                windowed_cross_attention_mask.view(
                    logits.size(0), logits.size(1)
                )
                .sum(dim=-1)
                .max()
                .item()
            )
            logits = logits[:, :max_len, :, :].contiguous()

        audio_loss = 0.0
        if new_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            for codeboox_idx in range(new_labels.size(-1)):
                # add valid_indices filtering
                codebook_loss = loss_fct(
                    logits[..., codeboox_idx, :].view(
                        -1,
                        (self.codec_decoder.codebook_size),
                    ),
                    new_labels[..., codeboox_idx].contiguous().view(-1),
                )
                audio_loss += (
                    codebook_loss * self.codebook_weights[codeboox_idx]
                )

            predicted_labels = sample_token(
                logits.float(),
                self.use_sampling,
                self.temp_audio,
                self.top_k_audio,
            )  # B, T, codebook_size

            # compute accuracy and top10 accuracy for each codebook
            metrics_per_codebook = {}
            _, topk_indices = torch.topk(logits.softmax(dim=-1), 10, dim=-1)
            expanded_labels = new_labels.unsqueeze(-1).expand_as(topk_indices)
            correct_predictions = (topk_indices == expanded_labels).any(dim=-1)
            for codebook_idx in range(new_labels.size(-1)):
                # compute accuracy
                p = predicted_labels.contiguous()[..., codebook_idx].view(-1)[
                    valid_indices.contiguous()[..., codebook_idx].view(-1)
                ]
                g = new_labels.contiguous()[..., codebook_idx].view(-1)[
                    valid_indices.contiguous()[..., codebook_idx].view(-1)
                ]
                # compute top10 accuracy
                valid_dim_indices = valid_indices[:, :, codebook_idx]
                correct_dim_predictions = correct_predictions[
                    :, :, codebook_idx
                ]
                top10accuracy = (
                    correct_dim_predictions[valid_dim_indices].float().mean()
                )
                # store metrics
                metrics_per_codebook[codebook_idx] = {
                    "accuracy": (p == g).float().mean(),
                    "top10_accuracy": top10accuracy.item(),
                }
        audio_loss = audio_loss / self.codec_decoder.num_quantizers
        audio_loss = audio_loss * self.inverse_decay_fn(
            self.config.audio_loss_weight,
            self.trainer.state.global_step / self.trainer.state.max_steps,
            self.config.audio_loss_decay,
        )
        return audio_loss, metrics_per_codebook

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        audios: Optional[List[Tuple[torch.Tensor, int]]] = None,
        condition_audios: Optional[List[Tuple[torch.Tensor, int]]] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        videos: Optional[List[Tuple[torch.Tensor, int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if audios is not None and images is not None:
            raise ValueError(
                "`generate()` currently doesn't support both `audios` and `images`."
            )

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        tts_max_len = kwargs.pop("tts_max_len", 5)
        streamer = kwargs.pop("streamer", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)

        is_input_multimodal = (
            audios is not None
            or videos is not None
            or condition_audios is not None
            or images is not None
        )
        if is_input_multimodal:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
            ) = self.prepare_multimodal_inputs_labels(
                text_inputs={
                    "input_ids": inputs,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": None,
                    "labels": None,
                },
                modalities_args={
                    "audio": {
                        "input_audios_srs": audios,
                        "condition_audios_srs": condition_audios,
                        "transcription_ids": None,
                        "transcription_attention_mask": None,
                    },
                    "image": {
                        "images": images,
                        "image_sizes": (
                            image_sizes if images is not None else None
                        ),
                    },
                    "video": {
                        "input_videos_srs": videos,
                        "video_transcription_ids": None,
                        "video_transcription_attention_mask": None,
                    },
                },
            )
        else:
            inputs_embeds = self.text_decoder.model.get_input_embeddings()(
                inputs
            )

        if self.talking_head is not None:
            return self._generate_with_talking_head(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                hardcoded_start_audio_token=False,
                tts_max_len=tts_max_len,
                **kwargs,
            )

        return self.text_decoder.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            streamer=streamer,
            **kwargs,
        )

    @torch.no_grad()
    def _generate_with_talking_head(
        self,
        position_ids,
        attention_mask,
        inputs_embeds,
        tts_max_len=5,
        hardcoded_start_audio_token=False,
        **kwargs,
    ):
        audio_values = None
        tts_max_len = (
            tts_max_len * self.codec_decoder.input_sampling_rate
        )  # seconds to frames conversion
        tokenizer = self.text_decoder.tokenizer
        is_moshi_like = isinstance(self.talking_head, MoshiBertTalkingHead)
        tts_input = kwargs.pop("tts_input", None)
        force_text_tokens = (
            kwargs.pop("force_text_tokens", False)
            and tts_input is not None
            and tts_input != ""
            and is_moshi_like
        )
        acoustic_delay = getattr(self.config, "acoustic_delay", 0)

        B, S = inputs_embeds.shape[:2]
        assert B == 1, "Batch size must be 1"

        self.text_decoder.eval()
        self.talking_head.eval()
        codelist, output_ids, audio_paths = [], [], []
        start_of_audio = True if hardcoded_start_audio_token else False

        gt_text_tokens = None
        audio_values = None
        if force_text_tokens:
            # force the model with the provided text at start and when predicting token different from pad_audio_token_id and epad_audio_token_id
            words = re.findall(r"\S+\s*", tts_input)
            gt_text_tokens = []
            word_counter = 0
            token_offset = 0
            no_epad_counter = 0
            for w in words:
                gt_text_tokens.append(tokenizer.encode(w)[1:])

        with torch.inference_mode():
            step = -1
            step_tts = -1
            while True:
                if not start_of_audio:
                    step += 1
                outputs = self.text_decoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=kwargs.pop("past_key_values", None),
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=kwargs.pop("output_attentions", False),
                    output_hidden_states=kwargs.pop(
                        "output_hidden_states", True
                    ),
                    return_dict=kwargs.pop("return_dict", True),
                    cache_position=kwargs.pop("cache_position", None),
                )

                hidden_states = outputs.hidden_states[-1][:, -1, :].squeeze(0)

                text_logits = self.text_decoder.get_output_embeddings()(
                    hidden_states
                )
                text_token = sample_token(
                    text_logits.float(),
                    use_sampling=True,
                    temp=self.temp_text,  # 0.7
                    top_k=self.top_k_text,  # 25
                )
                if text_token == self.text_decoder.tokenizer.eos_token_id:
                    break
                elif (
                    text_token
                    == self.text_decoder.tokenizer.get_added_vocab()[
                        DEFAULT_AUDIO_OUTPUT_START_TOKEN
                    ]
                    and not start_of_audio
                ):
                    print("START OF AUDIO")
                    start_of_audio = True
                    output_ids.append(text_token)
                    next_token = (
                        self.text_decoder.model.get_input_embeddings()(
                            text_token
                        )
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )
                    inputs_embeds = torch.cat(
                        [inputs_embeds, next_token.to(inputs_embeds.device)],
                        dim=1,
                    )
                    if acoustic_delay:
                        cur_audio_frame_step = 0
                    continue
                elif (
                    text_token
                    == self.text_decoder.tokenizer.get_added_vocab()[
                        DEFAULT_AUDIO_OUTPUT_END_TOKEN
                    ]
                    or step_tts >= tts_max_len
                ) and start_of_audio:
                    if (
                        text_token
                        == self.text_decoder.tokenizer.get_added_vocab()[
                            DEFAULT_AUDIO_OUTPUT_END_TOKEN
                        ]
                    ):
                        output_ids.append(text_token)
                        print("END OF AUDIO")

                    start_of_audio = False

                    # save the audio
                    codes = None  # B, K, S
                    if len(codelist) > 0:
                        codes = torch.cat(codelist, dim=1)
                        print(codes)
                        codes = codes.permute(0, 2, 1)  # B, K, S
                    if codes is None:
                        print("NO CODES")
                        break
                    audio_values = self.codec_decoder.model.decode(
                        codes
                    ).audio_values

                    if step_tts >= tts_max_len:
                        print(f"MAX LEN REACHED ({tts_max_len})")
                        break

                    break  # only generate one audio at a time
                    prev_codes = None
                    codelist = []
                    output_ids.append(text_token)
                    next_token = (
                        sself.text_decoder.model.get_input_embeddings()(
                            text_token
                        )
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )
                    inputs_embeds = torch.cat(
                        [inputs_embeds, next_token.to(inputs_embeds.device)],
                        dim=1,
                    )

                elif (
                    is_moshi_like
                    and text_token
                    != self.text_decoder.tokenizer.get_added_vocab()[
                        DEFAULT_AUDIO_PAD_TOKEN
                    ]
                    and text_token
                    != self.text_decoder.tokenizer.get_added_vocab()[
                        DEFAULT_AUDIO_EPAD_TOKEN
                    ]
                ):
                    pass
                    # print(f"token generated: {tokenizer.decode([text_token])}")

                output_ids.append(text_token)
                if step >= kwargs.get("max_new_tokens", 100):
                    print(f"MAX LEN REACHED ({tts_max_len})")
                    break
                next_token = (
                    self.text_decoder.model.get_input_embeddings()(text_token)
                    .unsqueeze(0)
                    .unsqueeze(1)
                )

                if start_of_audio:
                    step_tts += 1
                    if force_text_tokens:
                        (
                            forced_token_tensor,
                            word_counter,
                            token_offset,
                            no_epad_counter,
                        ) = self.get_forced_token(
                            tokenizer,
                            gt_text_tokens,
                            word_counter,
                            token_offset,
                            no_epad_counter,
                            inputs_embeds.device,
                        )
                        print(
                            f"forced token: {tokenizer.decode([forced_token_tensor])}"
                        )
                        output_ids[-1] = forced_token_tensor

                    if is_moshi_like:
                        codes = self.talking_head.generate(
                            context_vector=hidden_states.unsqueeze(
                                0
                            ).unsqueeze(1),
                            text_token=text_token.unsqueeze(0).unsqueeze(1),
                            sample=True,
                            temperature=self.temp_audio,
                            top_k=self.top_k_audio,
                            semantic_only=(
                                cur_audio_frame_step < acoustic_delay
                                if acoustic_delay
                                else False
                            ),
                        )
                    else:
                        codes = self.talking_head.generate(
                            context_vector=hidden_states.unsqueeze(
                                0
                            ).unsqueeze(1),
                            context_attention_mask=torch.ones(1, 1)
                            .bool()
                            .to(hidden_states.device),
                            sample=True,
                            temperature=self.temp_audio,
                            top_k=self.top_k_audio,
                        )
                    if acoustic_delay:
                        if cur_audio_frame_step >= acoustic_delay:
                            codelist.append(codes)
                        cur_audio_frame_step += 1
                    else:
                        codelist.append(codes)

                    # FIXME(st3p99): handle different self.backfeeding_audio_adapter
                    if isinstance(
                        self.backfeeding_audio_adapter,
                        BackfeedingAdapterOnFeatures,
                    ) and isinstance(
                        self.backfeeding_audio_adapter.adapter,
                        (MlpAdapter, WindowLevelQformerAdapter),
                    ):
                        if isinstance(
                            self.backfeeding_audio_adapter.adapter, MlpAdapter
                        ):
                            speech_projector_kwargs = {
                                "text_embeddings": self.text_decoder.model.get_input_embeddings()(
                                    text_token
                                ),
                                "training": False,
                            }
                        elif isinstance(
                            self.backfeeding_audio_adapter.adapter,
                            WindowLevelQformerAdapter,
                        ):
                            speech_projector_kwargs = {
                                "text_embeddings": self.text_decoder.model.get_input_embeddings()(
                                    text_token
                                ),
                                "training": False,
                            }
                            # speech_projector_kwargs = {
                            #     "text_embeddings": self.text_decoder.model.get_input_embeddings()(
                            #         torch.tensor(self.text_decoder.tokenizer.get_added_vocab()[DEFAULT_AUDIO_PAD_TOKEN]).to(text_token.device)
                            #     ).unsqueeze(0).unsqueeze(0),
                            #     "training": False,
                            # }
                        features = self.codec_encoder.get_features_from_codes(
                            codes
                        ).to(
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        )
                        audio_encoder_output = CodecOutput(
                            codes=codes,
                            features=features,
                            padding_mask=torch.ones(
                                features.shape[:2],
                                dtype=torch.bool,
                                device=features.device,
                            ),
                        )
                        next_token = self.backfeeding_audio_adapter(
                            audio_encoder_output,
                            **speech_projector_kwargs,
                        ).features
                    else:
                        raise NotImplementedError(
                            "Only BackfeedingAdapterOnFeatures with MLPAdapter is supported"
                        )

                inputs_embeds = torch.cat(
                    [inputs_embeds, next_token.to(inputs_embeds.device)], dim=1
                )

        return (audio_values, output_ids)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        inputs = self.get_text_decoder().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if "images" in kwargs:
            inputs["images"] = kwargs.pop("images")
        if "image_sizes" in kwargs:
            inputs["image_sizes"] = kwargs.pop("image_sizes")
        if "audios" in kwargs:
            inputs["audios"] = kwargs.pop("audios")
        if "videos" in kwargs:
            inputs["videos"] = kwargs.pop("videos")
        return inputs

    def encode_images(self, images):
        image_features = self.vision_encoder(images)
        image_features = self.vision_adapter(image_features)
        return image_features

    def encode_audios(self, audios):
        return self.audio_encoder(audios)

    def backfeeding_encode_audios(self, audios):
        return self.backfeeding_audio_adapter(audios)

    def adapt_encoded_audio(self, audio_features, audio_adapter_kwargs=None):
        return self.audio_adapter(
            audio_features, **(audio_adapter_kwargs or {})
        )

    def encode_audio(self, type: str, audios: List):
        if type == "input":
            return self.audio_encoder(audios)
        elif type == "condition":
            return self.codec_encoder(audios)
        elif type == "output":
            cur_step_max_step_ratio = (
                self.trainer.state.global_step / self.trainer.state.max_steps
            )
            return self.codec_encoder(
                audios,
                perturb_prob=self.decay_fn(
                    self.config.perturb_prob,
                    cur_step_max_step_ratio,
                    self.config.perturb_prob_decay,
                ),
            )
        else:
            raise ValueError(
                f"Audio modality type {type} not supported. Only 'input', 'condition' and 'output' are supported."
            )

    def adapt_audio(
        self,
        type: str,
        audio_module_output: SpeechLmmModuleOutput,
        audio_adapter_kwargs: dict = None,
    ):
        """
        type: str [input, condition, output]
        """
        if type == "input":
            adapter = getattr(self, f"audio_adapter")
        elif type == "condition":
            adapter = getattr(self, f"conditioning_audio_adapter")
        elif type == "output":
            adapter = getattr(self, f"backfeeding_audio_adapter")
        if adapter is None:
            raise ValueError(f"Adapter {adapter} not found.")
        adapted_output = adapter(
            audio_module_output, **(audio_adapter_kwargs or {})
        )
        if type == "condition" and getattr(
            adapter.config, "triplet_loss", False
        ):
            triplet_loss = self.compute_triplet_loss(adapted_output)
            adapted_output.triplet_loss = triplet_loss
        return adapted_output

    def compute_triplet_loss(self, adapted_output: SpeechLmmModuleOutput):
        B, T, _ = adapted_output.features.size()
        tokens_per_adapted_frame = getattr(
            self.conditioning_audio_adapter.config, "num_queries", 1
        )
        audio_features = adapted_output.features.view(
            B,
            -1,
            tokens_per_adapted_frame,
            adapted_output.features.size(-1),
        )
        attention_mask = ~(
            adapted_output.padding_mask.view(B, -1, tokens_per_adapted_frame)
        )
        losses = []
        triplet_loss = nn.TripletMarginLoss(
            margin=1.0,
            p=2,
            eps=1e-7,
        )
        for token_idx in range(tokens_per_adapted_frame):
            for batch_idx in range(B):
                cur_audio_features = audio_features[batch_idx][
                    attention_mask[batch_idx]
                ].view(-1, tokens_per_adapted_frame, audio_features.size(-1))[
                    :, token_idx, :
                ]
                for seq_idx, anchor in enumerate(cur_audio_features):
                    # positive
                    positive_mask = (
                        torch.zeros(audio_features.shape[:3])
                        .bool()
                        .to(device=audio_features.device)
                    )
                    positive_mask[batch_idx, :, token_idx] = True
                    positive_mask[batch_idx, seq_idx, token_idx] = False
                    # negative
                    negative_mask = ~positive_mask
                    # consider the original attention mask
                    positive_mask *= attention_mask
                    negative_mask *= attention_mask
                    # positive and negative samples
                    positive = audio_features[positive_mask].view(
                        -1, audio_features.size(-1)
                    )
                    negative = audio_features[negative_mask].view(
                        -1, audio_features.size(-1)
                    )
                    min_len = min(positive.shape[0], negative.shape[0])
                    positive = positive[:min_len]
                    negative = negative[:min_len]
                    loss = triplet_loss(
                        anchor.unsqueeze(0), positive, negative
                    )
                    losses.append(loss)
        return torch.tensor(losses).mean()

    def get_position_of_multimodal_features(self, token_ids: torch.LongTensor):
        """
        token_ids: a single sample in the batch
        """
        modalities = self.modalities
        count_dict = dict(zip(modalities, [0] * len(modalities)))
        ordered_modalities = []
        starts, ends = [], []
        for start_placeholder, end_placeholder, modality in zip(
            self.start_tokens, self.end_tokens, modalities
        ):
            start = torch.where(token_ids == start_placeholder)[0]
            end = torch.where(token_ids == end_placeholder)[0]
            assert (
                start.shape == end.shape
            ), f"# start and # end placeholders must match, found {start.shape[0]} for start and {end.shape[0]} for end"
            start = start + 1 if modality == "audio_output" else start
            starts.append(start)
            ends.append(end)
            # update number of sample per modality
            count_dict[modality] = start.shape[0]

            ordered_modalities += [modality] * count_dict[
                modality
            ]  # add n times the modality
        if len(starts) == 0:  # only text
            return (
                torch.tensor([[0, token_ids.shape[0]]]),
                ["none"],
                count_dict,
            )
        start = torch.cat(starts).to(device="cpu")
        end = torch.cat(ends).to(device="cpu")
        assert torch.equal(
            start.sort().indices, end.sort().indices
        ), "sort error, incoherent start and end tokens"
        modalities = np.atleast_1d(
            np.array(ordered_modalities)[start.sort().indices]
        ).tolist()
        # sort start and end
        start, end = start.sort().values, end.sort().values

        start = torch.cat(
            [
                start + 1,
                torch.tensor(token_ids.shape[0]).unsqueeze(0).to(device="cpu"),
            ]
        ).long()
        end = torch.cat([torch.zeros(1).to(device="cpu"), end]).long()
        text_ranges = torch.stack([end, start]).T

        if len(text_ranges) == len(modalities) + 1:
            # NOTE(st3p99): if token_ids ends with text tokens
            modalities.append("none")

        return text_ranges, modalities, count_dict

    def insert_multimodal_features(
        self,
        cur_text_embeds: torch.FloatTensor,
        cur_input_ids: torch.LongTensor,
        cur_labels: torch.LongTensor,
        audio_features: dict,
        image_features: torch.FloatTensor,
        video_features: dict,
        batch_idx: int,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        cur_text_embeds: single sample of the batch of text embeddings
        cur_labels: single sample of the batch of labels
        audio_features: dict of all the audio features - {
            'audio_input': {
                'feature': List[torch.Tensor], 'attention_mask': List[torch.BoolTensor]
                },
            'audio_condition': {
                'feature': List[torch.Tensor], 'attention_mask': List[torch.BoolTensor]
                },
            'audio_output': {
                'feature': List[torch.Tensor], 'attention_mask': List[torch.BoolTensor]
                }
        }
        image_features: all the image features
        video_features: dict of all the video features - {
            'video_input': {
                'feature': List[torch.Tensor], 'attention_mask': List[torch.BoolTensor]
                }
        }
        return the new_input_embeds and the new_labels,
        having the multimodal features inserted,
        with shape: (n_tokens, hidden_size), (n_tokens,)
        """
        text_ranges, modalities, n_feature_per_modality = (
            self.get_position_of_multimodal_features(cur_input_ids)
        )
        cur_new_input_embeds = []
        cur_new_labels = []
        count_dict = dict(
            zip(
                n_feature_per_modality.keys(),
                [0] * len(n_feature_per_modality.keys()),
            )
        )  # init counter per each modality
        for (start, end), modality in zip(text_ranges, modalities):
            # Add text embeddings
            cur_new_input_embeds.append(cur_text_embeds[start:end, :])
            cur_new_labels.append(cur_labels[start:end])
            if (
                modality.startswith("audio")
                and count_dict[modality] < n_feature_per_modality[modality]
            ):
                # Adding multimodal embeddings after text embeddings
                cur_features = audio_features[modality]["feature"][
                    batch_idx + count_dict[modality]
                ]
                cur_attn_mask = audio_features[modality]["attention_mask"][
                    batch_idx + count_dict[modality]
                ]
                if modality == "audio_output":
                    # cut = getattr(self.backfeeding_audio_adapter.adapter.config,"num_queries", 1)
                    cur_attn_mask = cur_attn_mask[1:]  # remove the first token
                audio_to_append = cur_features[cur_attn_mask].to(self.device)
                cur_new_input_embeds.append(audio_to_append)
                if modality == "audio_output":
                    # NOTE (pier): the reason of this check is because
                    # padding is not requested in case of audio output,
                    # for the labels are already provided by the dataloader
                    cur_new_labels.append(
                        cur_labels[end : end + audio_to_append.shape[0]]
                    )
                    assert (
                        cur_labels[end - 2]
                        == self.text_decoder.tokenizer.get_added_vocab()[
                            DEFAULT_AUDIO_OUTPUT_START_TOKEN
                        ]
                    )
                    assert (
                        cur_labels[end + audio_to_append.shape[0]]
                        == self.text_decoder.tokenizer.get_added_vocab()[
                            DEFAULT_AUDIO_OUTPUT_END_TOKEN
                        ]
                    )
                    assert (
                        cur_new_labels[-1][-1]
                        != self.text_decoder.tokenizer.get_added_vocab()[
                            DEFAULT_AUDIO_OUTPUT_END_TOKEN
                        ]
                    )
                else:
                    cur_new_labels.append(
                        torch.full(
                            (audio_to_append.shape[0],),
                            IGNORE_INDEX,
                            device=self.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            elif (
                modality.startswith("image")
                and count_dict[modality] < n_feature_per_modality[modality]
            ):
                cur_features = image_features[modality]["feature"][
                    batch_idx + count_dict[modality]
                ]
                cur_attn_mask = image_features[modality]["attention_mask"][
                    batch_idx + count_dict[modality]
                ]
                image_to_append = cur_features[cur_attn_mask].to(self.device)
                cur_new_input_embeds.append(image_to_append)
                cur_new_labels.append(
                    torch.full(
                        (image_to_append.shape[0],),
                        IGNORE_INDEX,
                        device=self.device,
                        dtype=cur_labels.dtype,
                    )
                )
            elif (
                modality.startswith("video")
                and count_dict[modality] < n_feature_per_modality[modality]
            ):
                cur_features = video_features[modality]["feature"][
                    batch_idx + count_dict[modality]
                ]
                cur_attn_mask = video_features[modality]["attention_mask"][
                    batch_idx + count_dict[modality]
                ]
                video_to_append = cur_features[cur_attn_mask].to(self.device)
                cur_new_input_embeds.append(video_to_append)
                cur_new_labels.append(
                    torch.full(
                        (video_to_append.shape[0],),
                        IGNORE_INDEX,
                        device=self.device,
                        dtype=cur_labels.dtype,
                    )
                )
            elif modality == "none":
                continue
            else:
                raise ValueError(f"Unexpected modality: {modality}")
            count_dict[modality] += 1

        for modality, count in count_dict.items():
            assert (
                count == n_feature_per_modality[modality]
            ), f"Expected {n_feature_per_modality[modality]} features for {modality}, found {count}"

        cur_new_input_embeds = torch.cat(
            cur_new_input_embeds
        )  # shape: (n_tokens, hidden_size)
        cur_new_labels = torch.cat(cur_new_labels)  # shape: (n_tokens,)

        return cur_new_input_embeds, cur_new_labels

    def encode_videos(self, videos, video_adapter_kwargs=None):
        video_encoder_output = self.video_encoder(videos)
        video_adapter_output = self.video_adapter(
            video_encoder_output, **(video_adapter_kwargs or {})
        )
        return video_adapter_output

    def prepare_multimodal_inputs_labels(
        self,
        text_inputs: dict,
        modalities_args: dict,
    ):
        """
        Prepare inputs for multiple modalities and their respective labels.
        The 'text_inputs' argument contains core text-based inputs.
        The 'modalities_args' argument is a dictionary containing modality-specific inputs.
        Each modality key (e.g., 'audio', 'image', 'video', etc.) has its own set of parameters.

        Example usage:
        prepare_multimodal_inputs_labels(
                text_inputs={
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "labels": labels
                },
                modalities_args={
                    "audio": {
                        "audios_srs": audios_srs,
                        "transcription_ids": transcription_ids,
                        "transcription_attention_mask": transcription_attention_mask,
                        },
                    "image": {
                        "images": images,
                        "image_sizes": image_sizes,
                    },
                    "video": {
                        "input_videos_srs": input_videos_srs,
                        "transcription_ids": transcription_ids,
                        "transcription_attention_mask": transcription_attention_mask,
                    },
                },
            )
        """
        # Text-based inputs
        input_ids = text_inputs.get("input_ids")
        position_ids = text_inputs.get("position_ids")
        attention_mask = text_inputs.get("attention_mask")
        past_key_values = text_inputs.get("past_key_values", None)
        labels = text_inputs.get("labels", None)
        raw_audio_adapter_output = None
        raw_video_adapter_output = None

        self.start_tokens = []
        self.end_tokens = []
        self.modalities = []

        # empty features
        audio_features = {
            "audio_input": None,
            "audio_condition": None,
            "audio_output": None,
        }
        video_features = {
            "video_input": None,
        }
        image_features = {
            "image_input": None,
        }

        # Process modality-specific inputs
        # ----------------------------------------
        no_modalities = True
        if "audio" in modalities_args and (
            modalities_args["audio"].get("input_audios_srs") is not None
            or modalities_args["audio"].get("output_audios_srs") is not None
            or modalities_args["audio"].get("condition_audios_srs") is not None
        ):
            no_modalities = False
            # Extract audio-related inputs
            audio_inputs = modalities_args["audio"]
            input_audios_srs = audio_inputs.get("input_audios_srs")
            output_audios_srs = audio_inputs.get("output_audios_srs", None)
            condition_audios_srs = audio_inputs.get(
                "condition_audios_srs", None
            )
            transcription_ids = audio_inputs.get("transcription_ids", None)
            transcription_attention_mask = audio_inputs.get(
                "transcription_attention_mask", None
            )
            aligned_transcription_ids = audio_inputs.get(
                "aligned_transcription_ids", None
            )

            raw_audio_adapter_output = {
                "audio_input": None,
                "audio_condition": None,
                "audio_output": None,
            }

            # Proces Assistant audio features if present (AUDIO IN OUTPUT)
            # ---------------------------------------------------------------
            # backfeeding_audio_adapter -> audio adapter output
            for audios_srs, audio_type in zip(
                [input_audios_srs, condition_audios_srs, output_audios_srs],
                ["input", "condition", "output"],
            ):
                # Filter out None entries in the audios_srs list
                if audios_srs is None:
                    continue

                valid_audio_indices = [
                    i
                    for i, audio in enumerate(audios_srs)
                    if audio is not None
                ]
                valid_audios = [
                    audio for audio in audios_srs if audio is not None
                ]
                if audio_type == "input" and transcription_ids is not None:
                    valid_transcription_ids = [
                        (transcription_ids[i], transcription_attention_mask[i])
                        for i in valid_audio_indices
                        if transcription_ids[i] is not None
                    ]
                    transcription_ids = (
                        torch.nn.utils.rnn.pad_sequence(
                            [x[0] for x in valid_transcription_ids],
                            batch_first=True,
                            padding_value=self.text_decoder.tokenizer.pad_token_id,
                        )
                        if valid_transcription_ids is not None
                        and len(valid_transcription_ids) > 0
                        else None
                    )
                    if transcription_ids is not None:
                        transcription_ids = transcription_ids[
                            :, : self.text_decoder.tokenizer.model_max_length
                        ]
                        transcription_attention_mask = (
                            torch.nn.utils.rnn.pad_sequence(
                                [x[1] for x in valid_transcription_ids],
                                batch_first=True,
                                padding_value=0,
                            )
                        )
                if (
                    audio_type == "output"
                    and aligned_transcription_ids is not None
                ):
                    valid_aligned_transcription_ids = [
                        aligned_transcription_ids[i]
                        for i in valid_audio_indices
                        if aligned_transcription_ids[i] is not None
                    ]
                    aligned_transcription_ids = (
                        torch.nn.utils.rnn.pad_sequence(
                            [x for x in valid_aligned_transcription_ids],
                            batch_first=True,
                            padding_value=self.text_decoder.tokenizer.pad_token_id,
                        )
                        if valid_aligned_transcription_ids is not None
                        and len(valid_aligned_transcription_ids) > 0
                        else None
                    )
                    aligned_transcription_ids = (
                        aligned_transcription_ids[
                            :, : self.text_decoder.tokenizer.model_max_length
                        ]
                        if aligned_transcription_ids is not None
                        else None
                    )
                    aligned_transcription_attention_mask = (
                        aligned_transcription_ids.ne(
                            self.text_decoder.tokenizer.pad_token_id
                        )
                    )

                if valid_audios is not None:
                    if audio_type == "input":
                        self.start_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_INPUT_START_TOKEN
                            ]
                        )
                        self.end_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_INPUT_END_TOKEN
                            ]
                        )
                        self.modalities.append("audio_input")
                    elif audio_type == "condition":
                        self.start_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_CONDITION_START_TOKEN
                            ]
                        )
                        self.end_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_CONDITION_END_TOKEN
                            ]
                        )
                        self.modalities.append("audio_condition")
                    elif audio_type == "output":
                        self.start_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_OUTPUT_START_TOKEN
                            ]
                        )
                        self.end_tokens.append(
                            self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_OUTPUT_END_TOKEN
                            ]
                        )
                        self.modalities.append("audio_output")
                    else:
                        raise ValueError(
                            f"Unexpected audio type: {audio_type}"
                        )

                    # encode and adapt
                    audio_encoder_output = self.encode_audio(
                        audio_type, valid_audios
                    )
                    audio_adapter_kwargs = {}
                    if (
                        audio_type == "output"
                        and self.config.align_text_to_audio
                    ):  # MoshiBertTalkingHead + No Variable Compression -> align_text_audio (token, pad, pad, epad, token)
                        assert (
                            aligned_transcription_ids is not None
                            and aligned_transcription_attention_mask
                            is not None
                        )
                        (
                            input_ids,
                            labels,
                            attention_mask,
                            aligned_transcription_ids,
                            aligned_transcription_attention_mask,
                        ) = AlignCodesAndText.handle_text_audio_alignment(
                            input_ids,
                            labels,
                            attention_mask,
                            aligned_transcription_ids,
                            aligned_transcription_attention_mask,
                            audio_encoder_output,
                            pad_token_id=self.text_decoder.tokenizer.pad_token_id,
                            pad_audio_token_id=self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_PAD_TOKEN
                            ],
                            epad_audio_token_id=self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_EPAD_TOKEN
                            ],
                            bos_audio_token_id=self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_OUTPUT_START_TOKEN
                            ],
                            eos_audio_token_id=self.text_decoder.tokenizer.get_added_vocab()[
                                DEFAULT_AUDIO_OUTPUT_END_TOKEN
                            ],
                        )
                        audio_adapter_kwargs = {
                            "text_embeddings": self.text_decoder.model.get_input_embeddings()(
                                aligned_transcription_ids
                            ),
                        }
                    elif audio_type == "input" and (
                        isinstance(
                            self.audio_adapter, (CifAdapter, CtcAdapter)
                        )
                        or (
                            hasattr(self.audio_adapter, "length_adapter")
                            and isinstance(
                                self.audio_adapter.length_adapter,
                                (CifAdapter, CtcAdapter),
                            )
                        )
                    ):
                        if transcription_ids is None:
                            raise ValueError(
                                "Transcription IDs are required for CIF/CTC adapters"
                            )
                        if transcription_attention_mask is None:
                            raise ValueError(
                                "Transcription attention mask is required for CIF/CTC adapters"
                            )
                        # transcription_ids and transcription_attention_mask
                        audio_adapter_kwargs = (
                            {
                                "transcription_ids": transcription_ids,
                                "transcription_attention_mask": transcription_attention_mask,
                            },
                        )
                    elif (
                        audio_type == "output"
                        and not self.config.align_text_to_audio
                    ):
                        if getattr(
                            self.talking_head, "duration_prediction", False
                        ):
                            raise NotImplementedError(
                                "Duration prediction is not supported yet for NARTalkingHead"
                            )
                        else:
                            pad_token = (
                                self.text_decoder.tokenizer.get_added_vocab()[
                                    DEFAULT_AUDIO_PAD_TOKEN
                                ]
                            )
                            audio_adapter_kwargs = {
                                "text_embeddings": self.text_decoder.model.get_input_embeddings()(
                                    torch.tensor(pad_token)
                                    .to(self.device)
                                    .unsqueeze(0)
                                ),
                                "repeat_padding": True,
                                "perturb_prob": self.config.perturb_prob,
                                "codec_encoder": self.codec_encoder,
                            }
                    cur_raw_audio_adapter_output = self.adapt_audio(
                        audio_type,
                        audio_encoder_output,
                        audio_adapter_kwargs=audio_adapter_kwargs,
                    )
                    if (
                        audio_type == "output"
                        and not self.config.align_text_to_audio
                    ):
                        # align with unknown tokens
                        (
                            input_ids,
                            labels,
                            attention_mask,
                        ) = AlignWithUnkownTokens(
                            self.text_decoder.tokenizer
                        ).align(
                            input_ids,
                            labels,
                            attention_mask,
                            ~cur_raw_audio_adapter_output.adapted_padding_mask,
                            talking_head_use_text_tokens=True,  # FIXME(pier): hardcoded - if False labels are ignored
                        )

                    features = [None] * len(audios_srs)
                    masks = [None] * len(audios_srs)
                    for i, idx in enumerate(valid_audio_indices):
                        features[idx] = cur_raw_audio_adapter_output.features[
                            i
                        ]
                        cur_padding_mask = getattr(
                            cur_raw_audio_adapter_output,
                            "adapted_padding_mask",
                            None,
                        )
                        if cur_padding_mask is None:
                            cur_padding_mask = getattr(
                                cur_raw_audio_adapter_output,
                                "padding_mask",
                            )
                        masks[idx] = ~(cur_padding_mask[i])

                    audio_features[f"audio_{audio_type}"] = {
                        "feature": features,
                        "attention_mask": masks,
                    }

                    raw_audio_adapter_output[f"audio_{audio_type}"] = (
                        cur_raw_audio_adapter_output  # FIXME: append to a list
                    )

        if (
            "video" in modalities_args
            and modalities_args["video"].get("input_videos_srs") is not None
        ):
            no_modalities = False
            # Extract video-related inputs
            video_inputs = modalities_args["video"]
            input_videos_srs = video_inputs.get("input_videos_srs")

            video_transcription_ids = video_inputs.get(
                "video_transcription_ids", None
            )
            if video_transcription_ids is not None:
                video_transcription_ids = torch.nn.utils.rnn.pad_sequence(
                    video_transcription_ids,
                    batch_first=True,
                    padding_value=self.text_decoder.tokenizer.pad_token_id,
                )

            video_transcription_attention_mask = video_inputs.get(
                "video_transcription_attention_mask", None
            )
            if video_transcription_attention_mask is not None:
                video_transcription_attention_mask = (
                    torch.nn.utils.rnn.pad_sequence(
                        video_transcription_attention_mask,
                        batch_first=True,
                        padding_value=0,
                    )
                )

            # Add video tokens to tracking lists
            self.start_tokens.append(
                self.text_decoder.tokenizer.get_added_vocab()[
                    DEFAULT_VIDEO_INPUT_START_TOKEN
                ]
            )
            self.end_tokens.append(
                self.text_decoder.tokenizer.get_added_vocab()[
                    DEFAULT_VIDEO_INPUT_END_TOKEN
                ]
            )
            self.modalities.append("video_input")

            # Process video features
            raw_video_adapter_output = self.encode_videos(
                input_videos_srs,
                video_adapter_kwargs={
                    "transcription_ids": video_transcription_ids,
                    "transcription_attention_mask": video_transcription_attention_mask,
                },
                # # ↑ required by video projectors that need to access the
                # # textual transcription as well (such as CIF and CTC which
                # # need to compute the ctc_loss on the transcription)
            )

            video_features = {
                "video_input": {
                    "feature": raw_video_adapter_output.features,
                    "attention_mask": ~raw_video_adapter_output.padding_mask,
                }
            }

        if (
            "image" in modalities_args
            and modalities_args["image"].get("images") is not None
        ):
            no_modalities = False
            # Extract and process image-related inputs
            image_inputs = modalities_args["image"]

            # Example of image processing logic
            cur_image_features = self.process_image_features(image_inputs)

            image_features["image_input"]["feature"] = cur_image_features

            self.start_tokens.append(
                self.text_decoder.tokenizer.get_added_vocab()[
                    DEFAULT_IM_START_TOKEN
                ]
            )
            self.end_tokens.append(
                self.text_decoder.tokenizer.get_added_vocab()[
                    DEFAULT_IM_END_TOKEN
                ]
            )
            self.modalities.append("image_input")

        if no_modalities:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
                None,
            )

        # Combine processed inputs for final usage
        # ------------------------------------------------

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove padding using attention_mask, obtaining list of tensors
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(
                input_ids, attention_mask
            )
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # Initialize lists for processed inputs and labels
        new_input_embeds = []
        new_labels = []
        # Process modalities
        #
        audio_idx = 0
        image_idx = 0
        video_idx = 0
        for (
            batch_idx,
            cur_input_ids,
        ) in enumerate(input_ids):
            # check if multimodal tokens are present
            multimodal_input_present = len(self.start_tokens) > 0

            if not multimodal_input_present:
                new_input_embeds.append(
                    self.text_decoder.model.get_input_embeddings()(
                        cur_input_ids
                    )
                )
                new_labels.append(labels[batch_idx])
                continue

            # ---------------------- handling multimodal tokens ---------------------- #
            # get the embeddings only for text tokens

            cur_labels = labels[batch_idx]
            cur_text_embeds = self.text_decoder.model.get_input_embeddings()(
                cur_input_ids
            )

            # insert multimodal features
            cur_new_input_embeds, cur_new_labels = (
                self.insert_multimodal_features(
                    cur_text_embeds,
                    cur_input_ids,
                    cur_labels,
                    audio_features,
                    image_features,
                    video_features,
                    batch_idx,
                )
            )

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make
        # the sequence longer
        max_length = self.text_decoder.tokenizer.model_max_length
        if max_length is not None:
            new_input_embeds = [x[:max_length] for x in new_input_embeds]
            new_labels = [x[:max_length] for x in new_labels]

        # Stack them back as a single tensor, padding if necessary
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        # Pad the embeddings and labels
        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if self.config.tokenizer_padding_side == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:  # tokenizer_padding_side == "right"
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        new_labels = None if _labels is None else new_labels_padded
        attention_mask = (
            None
            if _attention_mask is None
            else attention_mask.to(dtype=_attention_mask.dtype)
        )
        position_ids = None if _position_ids is None else position_ids
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            raw_audio_adapter_output,
            raw_video_adapter_output,
        )

    def update_model_output_with_audio_adapter_output(
        self, model_output, audio_adapter_output
    ):
        model_output_dict = vars(model_output)
        lm_loss = model_output_dict.pop("loss")
        granular_losses = {"lm_loss": lm_loss}
        if isinstance(
            audio_adapter_output,
            (CifAdapterOutput, CtcAdapterOutput),
        ):
            if audio_adapter_output.ctc_loss is not None:
                if hasattr(self.config.audio_adapter, "length_adapter"):
                    # This is a two-stage adapter (length + modality)
                    ctc_loss_weight = (
                        self.config.audio_adapter.length_adapter.ctc_loss_weight
                    )
                else:
                    # This is CIF/CTC used as a standalone adapter
                    ctc_loss_weight = self.config.audio_adapter.ctc_loss_weight
                granular_losses["ctc_loss"] = (
                    audio_adapter_output.ctc_loss * ctc_loss_weight
                )
            if (
                isinstance(audio_adapter_output, CifAdapterOutput)
                and audio_adapter_output.quantity_loss is not None
            ):
                if hasattr(self.config.audio_adapter, "length_adapter"):
                    # This is a two-stage adapter (length + modality)
                    quantity_loss_weight = (
                        self.config.audio_adapter.length_adapter.quantity_loss_weight
                    )
                else:
                    # This is CIF used as a standalone adapter
                    quantity_loss_weight = (
                        self.config.audio_adapter.quantity_loss_weight
                    )
                granular_losses["quantity_loss"] = (
                    audio_adapter_output.quantity_loss * quantity_loss_weight
                )

        return CausalLMOutputWithPastAndGranularLosses(
            loss=sum(granular_losses.values()),
            granular_losses=granular_losses,
            **model_output_dict,
        )

    def update_model_output_with_talking_head_output(
        self, model_output, audio_loss, metrics_per_codebook
    ):
        model_output_dict = vars(model_output)
        loss = model_output_dict.pop("loss")

        granular_losses = model_output_dict.pop("granular_losses")
        granular_losses["audio_loss"] = audio_loss
        if granular_losses.get("lm_loss", None) is None:
            granular_losses["lm_loss"] = loss

        metrics = {}
        accuracy = 0.0
        top10_accuracy = 0.0
        assert len(metrics_per_codebook) == self.codec_decoder.num_quantizers
        for k, v in metrics_per_codebook.items():
            metrics[f"accuracy_{k}"] = v["accuracy"]
            metrics[f"top10_accuracy_{k}"] = v["top10_accuracy"]
            accuracy += v["accuracy"]
            top10_accuracy += v["top10_accuracy"]
        metrics["accuracy"] = accuracy / self.codec_decoder.num_quantizers
        metrics["top10_accuracy"] = (
            top10_accuracy / self.codec_decoder.num_quantizers
        )

        return CausalLMOutputWithPastAndGranularLossesAndMetrics(
            loss=sum(granular_losses.values()),
            granular_losses=granular_losses,
            metrics=metrics,
            **model_output_dict,
        )

    # TODO(anferico): this is almost 100% overlapping with `update_model_output_with_audio_adapter_output`
    def update_model_output_with_video_adapter_output(
        self, model_output, video_adapter_output
    ):
        model_output_dict = vars(model_output)
        lm_loss = model_output_dict.pop("loss")
        granular_losses = {"lm_loss": lm_loss}

        if isinstance(
            video_adapter_output,
            (CifAdapterOutput, CtcAdapterOutput),
        ):
            if video_adapter_output.ctc_loss is not None:
                if hasattr(self.config.video_adapter, "length_adapter"):
                    # This is a two-stage adapter (length + modality)
                    ctc_loss_weight = (
                        self.config.video_adapter.length_adapter.ctc_loss_weight
                    )
                else:
                    # This is CIF/CTC used as a standalone adapter
                    ctc_loss_weight = self.config.video_adapter.ctc_loss_weight
                granular_losses["ctc_loss"] = (
                    video_adapter_output.ctc_loss * ctc_loss_weight
                )
            if (
                isinstance(video_adapter_output, CifAdapterOutput)
                and video_adapter_output.quantity_loss is not None
            ):
                if hasattr(self.config.video_adapter, "length_adapter"):
                    # This is a two-stage adapter (length + modality)
                    quantity_loss_weight = (
                        self.config.video_adapter.length_adapter.quantity_loss_weight
                    )
                else:
                    # This is CIF used as a standalone adapter
                    quantity_loss_weight = (
                        self.config.video_adapter.quantity_loss_weight
                    )
                granular_losses["quantity_loss"] = (
                    video_adapter_output.quantity_loss * quantity_loss_weight
                )

        return CausalLMOutputWithPastAndGranularLosses(
            loss=sum(granular_losses.values()),
            granular_losses=granular_losses,
            **model_output_dict,
        )

    def get_forced_token(
        self,
        tokenizer,
        gt_text_tokens,
        word_counter,
        token_offset,
        no_epad_counter,
        device,
    ):
        """
        Determine and return the forced token for TTS inference.

        Args:
            tokenizer: The tokenizer instance
            gt_text_tokens: List of lists containing token ids for each word
            word_counter: Current word position
            token_offset: Current token position within the word
            no_epad_counter: Counter for time spent since last EPAD
            device: Device to place the tensor on

        Returns:
            tuple: (forced_token_tensor, new_word_counter, new_token_offset, new_no_epad_counter)
        """
        # Check timing for forcing EPAD (2 seconds)
        max_time_per_word = self.codec_encoder.output_sampling_rate * 2.0
        should_force_epad = no_epad_counter >= max_time_per_word

        # Determine which token to force
        if word_counter >= len(gt_text_tokens) or should_force_epad:
            # End of all words or timeout -> force EPAD
            forced_token = tokenizer.get_added_vocab()[
                DEFAULT_AUDIO_EPAD_TOKEN
            ]
        elif token_offset >= len(gt_text_tokens[word_counter]):
            # End of current word tokens -> force PAD
            forced_token = tokenizer.get_added_vocab()[DEFAULT_AUDIO_PAD_TOKEN]
        else:
            # Force next expected token for current word
            forced_token = gt_text_tokens[word_counter][token_offset]

        # Create tensor for forced token
        forced_token_tensor = torch.tensor(forced_token).to(device)

        # Update counters based on token type
        new_no_epad_counter = no_epad_counter
        new_word_counter = word_counter
        new_token_offset = token_offset

        if (
            forced_token
            == tokenizer.get_added_vocab()[DEFAULT_AUDIO_EPAD_TOKEN]
        ):
            # Reset counters and move to next word
            new_no_epad_counter = 0
            new_word_counter += 1
            new_token_offset = 0
        elif (
            forced_token
            != tokenizer.get_added_vocab()[DEFAULT_AUDIO_PAD_TOKEN]
        ):
            # Regular word token - move to next token position
            new_token_offset += 1
            new_no_epad_counter += 1
        else:
            # PAD token - just increment time counter
            new_no_epad_counter += 1

        return (
            forced_token_tensor,
            new_word_counter,
            new_token_offset,
            new_no_epad_counter,
        )
