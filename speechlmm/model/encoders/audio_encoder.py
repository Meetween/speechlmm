import contextlib
import logging
import math
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import einops
import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    EncodecConfig,
    EncodecModel,
    HubertModel,
    MimiConfig,
    MimiModel,
    PreTrainedModel,
    Wav2Vec2BertModel,
    WhisperConfig,
)
from transformers.cache_utils import Cache
from transformers.models.encodec.modeling_encodec import (
    EncodecDecoderOutput,
    EncodecEncoderOutput,
)
from transformers.models.mimi.modeling_mimi import (
    MimiDecoderOutput,
    MimiEncoderOutput,
)

from speechlmm.model.adapters.outputs import CodecOutput, SpeechLmmModuleOutput
from speechlmm.model.adapters.utils import (
    attention_weights_to_attention_mask,
    lengths_to_attention_mask,
)
from speechlmm.model.attn_implementation import AttentionImplementationMixin
from speechlmm.model.codestorm import CodeStorm
from speechlmm.model.utils import (
    compute_output_length_from_conv1d_hyperparams,
    compute_output_length_from_conv1d_layer,
)


def chunk_with_overlap(
    sequence, chunk_size: int, overlap: int = 0, drop_last: bool = False
):
    if len(sequence) <= chunk_size:
        return [sequence]

    hop_size = chunk_size - overlap
    chunks = []
    for i in range(0, len(sequence) - overlap, hop_size):
        chunk = sequence[i : i + chunk_size]
        if len(chunk) == chunk_size or not drop_last:
            chunks.append(chunk)
    return chunks


class HfAudioEncoder(
    AttentionImplementationMixin, torch.nn.Module, metaclass=ABCMeta
):
    """
    Base class for Hugging Face audio encoders. This class is a wrapper
    around Hugging Face audio encoders that provides a consistent interface for
    encoding audio features.

    Args:
        name_or_path (str, optional): The model name or path to load the model
          from. If not provided, it will be inferred from `config_kwargs`.
        delay_load (bool, optional): Whether to delay loading the model. If
          True, the model is not loaded until `_load_model` is called. Defaults
          to False.
        attn_implementation (str, optional): The attention implementation to
          use. If None, the default attention implementation from the model
          configuration is used. Defaults to None.
        torch_dtype (torch.dtype, optional): The data type to use for the
          model. If None, the default data type from the model configuration
          is used. Defaults to None.
        cache_dir (str, optional): The directory to cache pretrained weights.
        delay_load (bool, optional): Whether to delay loading the model. If
          True, the model is not loaded until `_load_model` is called. Defaults
          to False.
    """

    config_class = AutoConfig
    processor_class = AutoProcessor
    processor_audio_arg_name = None
    model_class = None
    model_forward_kwargs = {}

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        delay_load: bool = False,
        chunk_size_in_seconds: Optional[float] = None,
        chunk_overlap_in_seconds: float = 0.0,
        chunk_encoding_strategy: str = "loop",  # [batch, loop]
    ):
        if self.model_class is None or not issubclass(
            self.model_class, PreTrainedModel
        ):
            raise ValueError(
                f"Class attribute `model_class` must be a subclass of "
                f"`transformers.PreTrainedModel` (found {self.model_class})."
            )

        self._supports_flash_attn_2 = self.model_class._supports_flash_attn_2
        self._supports_sdpa = self.model_class._supports_sdpa

        super().__init__()

        config_dict = config_dict or {}
        name_or_path = name_or_path or config_dict.pop("_name_or_path", None)
        if name_or_path is None:
            raise ValueError(
                "`name_or_path` must be provided either as an explicit "
                "argument or as part of `config_kwargs` (in which case it "
                "should be named `_name_or_path`)."
            )

        torch_dtype_in_config = config_dict.pop("torch_dtype", None)
        torch_dtype = torch_dtype or torch_dtype_in_config
        self.config = self.config_class.from_pretrained(
            name_or_path, torch_dtype=torch_dtype, **config_dict
        )
        self.processor = self.processor_class.from_pretrained(name_or_path)
        self.name_or_path = name_or_path

        self.set_attn_implementation_with_fallback(attn_implementation)

        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.chunk_overlap_in_seconds = chunk_overlap_in_seconds
        self.chunk_encoding_strategy = chunk_encoding_strategy

        self.is_loaded = False
        if not delay_load:
            self._load_model()

    def _load_model(self):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.torch_dtype or default_dtype)
        self.model = self.model_class.from_pretrained(
            self.name_or_path,
            config=self.config,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
        )
        torch.set_default_dtype(default_dtype)
        self.is_loaded = True

    def forward(
        self, audios_srs: List[Tuple[torch.FloatTensor, int]]
    ) -> SpeechLmmModuleOutput:
        audios, sampling_rates = zip(*audios_srs)
        unique_sampling_rates = set(sampling_rates)
        if len(unique_sampling_rates) > 1:
            raise ValueError(
                "All audios must have the same sampling rate. "
                f"Found {len(unique_sampling_rates)} unique sampling rates: "
                f"{unique_sampling_rates}."
            )
        sr = unique_sampling_rates.pop()

        audios = [audio.squeeze().float().cpu().numpy() for audio in audios]
        original_batch_size = len(audios)

        need_chunking = self.chunk_size_in_seconds is not None
        if need_chunking:
            audios, n_chunks_per_audio = self._chunk_audios(audios, sr=sr)

        if need_chunking and self.chunk_encoding_strategy == "loop":
            chunkwise_features = []
            chunkwise_padding_masks = []
            # Encode audios using the same batch size specified in the
            # config, thus avoiding potential OOM errors
            for i in range(0, len(audios), original_batch_size):
                input_values, attention_mask = self._process_audios(
                    audios[i : i + original_batch_size], sr=sr
                )
                encoder_output = self._encode_processed_audios(
                    input_values, attention_mask
                )
                chunkwise_features.extend(
                    torch.unbind(encoder_output.features)
                )
                chunkwise_padding_masks.extend(
                    torch.unbind(encoder_output.padding_mask)
                )

            encoder_output = SpeechLmmModuleOutput(
                features=chunkwise_features,
                padding_mask=chunkwise_padding_masks,
            )
        else:  # ¬need_chunking | self.chunk_encoding_strategy == batch
            # if chunking is enabled, this means feeding the encoder
            # with batches of uncontrollable size (the batch size
            # depends on the number of chunks per audio), meaning that
            # we run the risk of incurring in OOM errors
            input_values, attention_mask = self._process_audios(audios, sr=sr)
            encoder_output = self._encode_processed_audios(
                input_values, attention_mask
            )

        if not need_chunking:
            return encoder_output

        return self._dechunk_output(encoder_output, n_chunks_per_audio)

    def _chunk_audios(self, audios, sr):
        chunk_size_in_samples = self.chunk_size_in_seconds * sr
        chunk_overlap_in_samples = self.chunk_overlap_in_seconds * sr
        chunked_audios = [
            chunk_with_overlap(
                audio,
                chunk_size=chunk_size_in_samples,
                overlap=chunk_overlap_in_samples,
                drop_last=False,
            )
            for audio in audios
        ]

        flattened_chunked_audios = [
            chunk for chunks in chunked_audios for chunk in chunks
        ]
        n_chunks_per_audio = [len(chunks) for chunks in chunked_audios]
        return flattened_chunked_audios, n_chunks_per_audio

    def _dechunk_output(
        self,
        encoder_output: SpeechLmmModuleOutput,
        n_chunks_per_audio: List[int],
    ) -> SpeechLmmModuleOutput:
        encoded_chunks: List[torch.FloatTensor] = encoder_output.features
        padding_mask_chunks: List[torch.BoolTensor] = (
            encoder_output.padding_mask
        )

        n_chunks_cumsum = np.cumsum(n_chunks_per_audio).tolist()
        chunk_starts = [0] + n_chunks_cumsum[:-1]
        chunk_ends = n_chunks_cumsum

        # regroup encoded chunks belonging to the same audio together
        dechunked_features = []
        dechunked_padding_masks = []
        for chunk_start_idx, chunk_end_idx in zip(chunk_starts, chunk_ends):
            *first_chunks, last_chunk = encoded_chunks[
                chunk_start_idx:chunk_end_idx
            ]
            *first_padding_masks, last_padding_mask = padding_mask_chunks[
                chunk_start_idx:chunk_end_idx
            ]
            # remove padding from last chunk (all the other chunks are
            # full max length chunks)
            SEQUENCE_LENGTH_DIM = 0
            # ↑ batch dimension was removed by the call to
            # `torch.unbind` in `forward`
            last_chunk = last_chunk[~last_padding_mask]
            dechunked_features.append(
                torch.cat(first_chunks + [last_chunk], dim=SEQUENCE_LENGTH_DIM)
            )
            last_padding_mask = last_padding_mask[~last_padding_mask]
            dechunked_padding_masks.append(
                torch.cat(
                    first_padding_masks + [last_padding_mask],
                    dim=SEQUENCE_LENGTH_DIM,
                )
            )

        features = torch.nn.utils.rnn.pad_sequence(
            dechunked_features,
            batch_first=True,
            padding_value=0.0,
        )
        padding_mask = torch.nn.utils.rnn.pad_sequence(
            dechunked_padding_masks,
            batch_first=True,
            padding_value=1.0,  # 1.0 = padding
        )

        return SpeechLmmModuleOutput(
            features=features, padding_mask=padding_mask
        )

    def _process_audios(self, audios: List[np.ndarray], sr: int, **kwargs):
        if sr != self.input_sampling_rate:
            raise ValueError(
                f"Sampling rate {sr} is not supported by this model. "
                f"Expected {self.input_sampling_rate}."
            )
        processor_args = []
        processor_kwargs = {
            "sampling_rate": sr,
            "return_tensors": "pt",
        }
        if self.processor_audio_arg_name is None:
            processor_args.append(audios)
        else:
            processor_kwargs[self.processor_audio_arg_name] = audios

        processor_output = self.processor(
            *processor_args, **processor_kwargs, **kwargs
        ).to(device=self.device, dtype=self.dtype)

        input_values_or_features = getattr(
            processor_output,
            "input_values",
            getattr(processor_output, "input_features", None),
        )
        if input_values_or_features is None:
            raise ValueError(
                "Expected `input_values` or `input_features` in the "
                "output of the processor."
            )

        attention_mask = getattr(
            processor_output,
            "attention_mask",
            getattr(processor_output, "padding_mask", None),
            # ↑ EnCodec returns `padding_mask` instead of
            # `attention_mask`. However, the `padding_mask` is
            # actually an attention mask 😅 Because of this, we
            # don't need to invert the mask
        )
        # NOTE: some processors do NOT return `attention_mask`
        # because the corresponding model was trained without it.
        # Instead, the model expects the input to be padded with 0s.

        return input_values_or_features, attention_mask.bool()

    def _encode_processed_audios(
        self, input_values, attention_mask: Optional[torch.BoolTensor] = None
    ) -> SpeechLmmModuleOutput:
        model_output = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=self.supports_output_attentions(),
            return_dict=True,
            **self.model_forward_kwargs,
        )

        if self.supports_output_attentions():
            attention_mask = attention_weights_to_attention_mask(
                model_output.attentions
            ).to(device=input_values.device)
        else:
            attention_mask = self._manually_recompute_attention_mask(
                attention_mask
            )

        return SpeechLmmModuleOutput(
            features=model_output.last_hidden_state,
            padding_mask=~attention_mask,
        )

    def supports_output_attentions(self) -> bool:
        return self.attn_implementation not in ["sdpa", "flash_attention_2"]

    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None
        raise NotImplementedError(
            f"Method `_manually_recompute_attention_mask` must be implemented "
            f"in {self.__class__.__qualname__} as "
            f"attn_implementation={self.attn_implementation} does not "
            f"support output_attentions=True."
        )

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def input_sampling_rate(self) -> int:
        with contextlib.suppress(AttributeError):
            return self.processor.sampling_rate
        with contextlib.suppress(AttributeError):
            return self.processor.feature_extractor.sampling_rate
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute "
            f"`input_sampling_rate`."
        )

    @property
    @abstractmethod
    def output_sampling_rate(self) -> float:
        """Sampling rate of the audio features output by the encoder."""
        return NotImplementedError(
            f"Property `output_sampling_rate` is not implemented in "
            f"{self.__class__.__name__}."
        )

    @property
    def hidden_size(self):
        return self.config.hidden_size


class HubertEncoder(HfAudioEncoder):
    processor_class = AutoFeatureExtractor
    processor_audio_arg_name = "raw_speech"
    model_class = HubertModel

    def _process_audios(self, audios: List[np.ndarray], sr: int, **kwargs):
        if (
            self.config.feat_extract_norm == "layer"
        ) != self.processor.return_attention_mask:
            raise ValueError(
                f"model.config.feat_extract_norm="
                f"{self.config.feat_extract_norm} and "
                f"processor.return_attention_mask="
                f"{self.processor.return_attention_mask} are not "
                f'consistent. feat_extract_norm="layer" should imply '
                f"return_attention_mask=True."
            )
            # NOTE: see comments below
        return super()._process_audios(
            audios,
            sr,
            padding=True,
            return_attention_mask=self.config.feat_extract_norm == "layer",
            # ↑ NOTE: from the Hugging Face documentation on Wav2Vec2
            # (HuBERT uses a Wav2Vec2 feature extractor):
            # Wav2Vec2 models that have set
            # `config.feat_extract_norm == "group"`, such as
            # wav2vec2-base, have **not** been trained using
            # `attention_mask`. For such models, `input_values` should
            # simply be padded with 0 and no `attention_mask`should be
            # passed.
            #
            # For Wav2Vec2 models that have set
            # `config.feat_extract_norm == "layer"`, such as
            # wav2vec2-lv60, `attention_mask` should be passed for
            # batched inference.
            # NOTE: also, from the Hugging Face documentation on HuBERT:
            # attention_mask should only be passed if the
            # corresponding processor has
            # config.return_attention_mask == True. For all models
            # whose processor has
            # config.return_attention_mask == False, such as
            # hubert-base, attention_mask should not be passed to
            # avoid degraded performance when doing batched
            # inference. For such models input_values should simply
            # be padded with 0 and passed without attention_mask.
            # Be aware that these models also yield slightly
            # different results depending on whether input_values
            # is padded or not.
            **kwargs,
        )

    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None

        pre_conv_attention_mask = attention_mask
        for conv_layer in self.model.feature_extractor.conv_layers:
            SEQ_LEN_DIM = 1
            pre_conv_lengths = pre_conv_attention_mask.sum(SEQ_LEN_DIM)
            post_conv_lengths = compute_output_length_from_conv1d_layer(
                pre_conv_lengths, conv1d_layer=conv_layer.conv
            )
            post_conv_attention_mask = lengths_to_attention_mask(
                post_conv_lengths.long()
            )
            # prepare for the next iteration
            pre_conv_attention_mask = post_conv_attention_mask

        return post_conv_attention_mask

    @property
    @lru_cache(maxsize=1)
    def output_sampling_rate(self) -> float:
        num_input_samples_in_100_seconds = self.input_sampling_rate * 100
        num_output_samples_in_100_seconds = num_input_samples_in_100_seconds
        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            num_output_samples_in_100_seconds = (
                compute_output_length_from_conv1d_hyperparams(
                    num_output_samples_in_100_seconds,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    dilation=1,
                )
            )

        return num_output_samples_in_100_seconds / 100


class Wav2Vec2BertEncoder(HfAudioEncoder):
    processor_class = AutoFeatureExtractor
    processor_audio_arg_name = "raw_speech"
    model_class = Wav2Vec2BertModel

    @property
    @lru_cache(maxsize=1)
    def output_sampling_rate(self) -> float:
        specgram_hop_length = 160
        # ↑ Hardcoded in the code used by Wave2Vec2Bert for computing
        # the Mel spectrogram
        num_input_samples_in_100_seconds = self.input_sampling_rate * 100
        num_output_samples_in_100_seconds = (
            num_input_samples_in_100_seconds / specgram_hop_length
        ) / self.processor.feature_extractor.stride
        if self.config.add_adapter:
            for _ in range(self.config.num_adapter_layers):
                num_output_samples_in_100_seconds = (
                    compute_output_length_from_conv1d_hyperparams(
                        num_output_samples_in_100_seconds,
                        kernel_size=self.config.adapter_kernel_size,
                        stride=self.config.adapter_stride,
                        padding=self.config.adapter_stride // 2,
                        dilation=1,
                    )
                )

        return num_output_samples_in_100_seconds / 100


class SeamlessM4Tv2Encoder(HfAudioEncoder):
    processor_audio_arg_name = "audios"
    model_class = (
        transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2SpeechEncoder
    )

    @property
    @lru_cache(maxsize=1)
    def output_sampling_rate(self) -> float:
        specgram_hop_length = 160
        # ↑ Hardcoded in the code used by SeamlessM4Tv2 for computing
        # the Mel spectrogram
        num_input_samples_in_100_seconds = self.input_sampling_rate * 100
        num_output_samples_in_100_seconds = (
            num_input_samples_in_100_seconds / specgram_hop_length
        ) / self.processor.feature_extractor.stride

        if self.config.add_adapter:
            for _ in range(self.config.num_adapter_layers):
                num_output_samples_in_100_seconds = (
                    compute_output_length_from_conv1d_hyperparams(
                        num_output_samples_in_100_seconds,
                        kernel_size=self.config.adaptor_kernel_size,
                        stride=self.config.adaptor_stride,
                        padding=self.config.adaptor_stride // 2,
                        dilation=1,
                    )
                )
                # ↑ NOTE: believe it or not, kernel size and stride are
                # misspelled in SeamlessM4Tv2's config (adaptor instead
                # of adapter, like in Wav2Vec2Bert)

        return num_output_samples_in_100_seconds / 100


class PatchedWhisperEncoder(
    transformers.models.whisper.modeling_whisper.WhisperEncoder
):
    # When loading model checkpoints, ignore the sinusoidal positional
    # encodings if they are missing from the checkpoint
    _sinusoidal_positional_encodings_key = "embed_positions.weight"
    _keys_to_ignore_on_load_missing = [_sinusoidal_positional_encodings_key]

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
        weights_only=True,
    ):
        # NOTE: we do the following because the original Whisper
        # checkpoint has a positional encoding layer with as many
        # sinusoidal embeddings as required by the maximum audio length
        # (30s), but in our case we want to reinstantiate the positional
        # encoding layer so that the maximum audio length is 10 minutes
        # (600s). Note that this is not an issue since the sinusoidal
        # embeddings are NOT learned, so it's safe to reinitialize them.
        # Still, before reinitializing them, we must remove them from
        # the state dict to keep Hugging Face transformers from
        # attempting to load them, which would raise an error due to
        # mismatched sizes
        if cls._sinusoidal_positional_encodings_key in state_dict:
            del state_dict[cls._sinusoidal_positional_encodings_key]
        if cls._sinusoidal_positional_encodings_key in loaded_keys:
            loaded_keys.remove(cls._sinusoidal_positional_encodings_key)

        return super()._load_pretrained_model(
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

    def _init_weights(self, module):
        # NOTE: prevents the error:
        #   TypeError: sinusoids() missing 1 required positional argument: 'channels'
        # when executing:
        #   embed_positions = module.embed_positions.weight
        #   embed_positions.copy_(sinusoids(*embed_positions.shape))
        # inside `WhisperPretrainedModel._init_weights()` while using
        # the Whisper audio encoder in combination with DeepSpeed ZeRO-3
        context_manager = (
            deepspeed.zero.GatheredParameters(
                module.embed_positions.weight, modifier_rank=0
            )
            if transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()
            and isinstance(module, PatchedWhisperEncoder)
            else contextlib.nullcontext()
        )
        with context_manager:
            super()._init_weights(module)


class BaseWhisperEncoder(HfAudioEncoder):
    processor_audio_arg_name = "audio"
    model_class = PatchedWhisperEncoder

    def _process_audios(self, audios: List[np.ndarray], sr: int, **kwargs):
        return super()._process_audios(
            audios,
            sr,
            device=self.device,  # Whisper can process audios on the GPU
            do_normalize=True,
            return_attention_mask=True,
            # ↑ NOTE: from the Hugging Face documentation on Whisper:
            # For Whisper models, attention_mask should always be passed
            # for batched inference, to avoid subtle bugs.
            **kwargs,
        )

    @property
    @lru_cache(maxsize=1)
    def output_sampling_rate(self) -> float:
        num_input_samples_in_100_seconds = self.input_sampling_rate * 100
        # Mel spectrogram
        num_output_samples_in_100_seconds = (
            num_input_samples_in_100_seconds
            / self.processor.feature_extractor.hop_length
        )
        # First convolutional layer
        num_output_samples_in_100_seconds = (
            compute_output_length_from_conv1d_hyperparams(
                num_output_samples_in_100_seconds,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            )
        )
        # Second convolutional layer
        num_output_samples_in_100_seconds = (
            compute_output_length_from_conv1d_hyperparams(
                num_output_samples_in_100_seconds,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
            )
        )

        return num_output_samples_in_100_seconds / 100


class WhisperEncoder(BaseWhisperEncoder):
    """
    Standard Whisper encoder with a fixed audio length of 30 seconds. Audios
    shorter than 30 seconds are padded to 30 seconds, and audios longer than 30
    seconds are truncated to 30 seconds.
    """

    # NOTE: although very questionable, this is effectively what Whisper
    # does internally in its `forward` method before passing its inputs
    # through the attention layers. In particular, it ignores the
    # `attention_mask` argument (it only cares about its shape),
    # resulting in attention weights that are > 0 even for masked
    # positions. The `attention_mask` argument is merely used to "avoid
    # performing SpecAugment data augmentation on padding token indices"
    # (source: https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel.forward)
    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None

        SEQ_LEN_DIM = 1
        pre_conv_length = attention_mask.shape[SEQ_LEN_DIM]
        post_conv1_length = compute_output_length_from_conv1d_layer(
            pre_conv_length, conv1d_layer=self.model.conv1
        )
        post_conv_length = compute_output_length_from_conv1d_layer(
            post_conv1_length, conv1d_layer=self.model.conv2
        )
        BATCH_SIZE_DIM = 0
        post_conv_attention_mask = attention_mask.new_ones(
            attention_mask.shape[BATCH_SIZE_DIM], post_conv_length
        )
        return post_conv_attention_mask


class UnrestrictedWhisperConfig(WhisperConfig):
    model_type = "whisper-unrestricted"


class UnrestrictedWhisperEncoder(BaseWhisperEncoder):
    """
    Modified Whisper encoder with an unrestricted audio length. Audios
    shorter than 30 seconds are NOT padded to 30 seconds, and audios longer
    than 30 seconds are NOT truncated to 30 seconds.
    """

    config_class = UnrestrictedWhisperConfig

    def _load_model(self):
        # The following modifications allow for audio lengths up to 10
        # minutes (600 seconds)
        self.processor.feature_extractor.chunk_length = 600
        self.processor.feature_extractor.n_samples = 9600000
        self.processor.feature_extractor.nb_max_frames = 60000
        self.config.max_source_positions = 30000

        super()._load_model()

    def supports_output_attentions(self):
        # The attention weights output by Whisper are flawed (they are
        # > 0 even for masked positions). By returning False, which
        # amounts to pretending that the model does not support
        # `output_attentions=True`, we trigger the manual computation of
        # proper attention weights in `_manually_recompute_attention_mask`
        return False

    # NOTE: this is *not* what Whisper does internally in its `forward`
    # method before passing its input through the attention layers, but
    # rather what it *should* do so that the `attention_mask` argument
    # is taken into account when computing the attention weights
    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        if self.supports_output_attentions():
            return None

        SEQ_LEN_DIM = 1
        pre_conv_lengths = attention_mask.sum(SEQ_LEN_DIM)
        post_conv1_lengths = compute_output_length_from_conv1d_layer(
            pre_conv_lengths, conv1d_layer=self.model.conv1
        )
        post_conv2_lengths = compute_output_length_from_conv1d_layer(
            post_conv1_lengths, conv1d_layer=self.model.conv2
        )
        post_conv_attention_mask = lengths_to_attention_mask(
            post_conv2_lengths
        )

        # All the audios in a batch are padded to the nearest integer
        # duration, so even the longest spectrogram might have some
        # padded frames
        is_longest_specgram_padded = (
            pre_conv_lengths.max() < attention_mask.shape[SEQ_LEN_DIM]
        )
        if is_longest_specgram_padded:
            max_padded_length = attention_mask.shape[SEQ_LEN_DIM]
            post_conv1_padded_length = compute_output_length_from_conv1d_layer(
                max_padded_length, conv1d_layer=self.model.conv1
            )
            post_conv2_padded_length = compute_output_length_from_conv1d_layer(
                post_conv1_padded_length, conv1d_layer=self.model.conv2
            )
            BATCH_SIZE_DIM = 0
            post_conv_attention_mask = torch.cat(
                [
                    post_conv_attention_mask,
                    torch.zeros(
                        post_conv_attention_mask.shape[BATCH_SIZE_DIM],
                        post_conv2_padded_length - post_conv2_lengths.max(),
                        dtype=torch.bool,
                        device=post_conv_attention_mask.device,
                    ),
                ],
                dim=SEQ_LEN_DIM,
            )

        return post_conv_attention_mask

    def _process_audios(self, audios: List[np.ndarray], sr: int, **kwargs):
        # Trick Whisper into thinking that the maximum audio length is
        # the maximum audio length in this particular batch of audios,
        # rounded up. We do this to prevent Whisper from arbitrarily
        # padding or truncating every input audio to a fixed length (30
        # seconds in the case of whisper-large-v3)
        old_chunk_length = self.processor.feature_extractor.chunk_length
        old_n_samples = self.processor.feature_extractor.n_samples
        old_nb_max_frames = self.processor.feature_extractor.nb_max_frames
        max_audio_length_in_samples = max(len(audio) for audio in audios)
        max_rounded_up_audio_length_in_seconds = math.ceil(
            max_audio_length_in_samples / sr
        )
        self.processor.feature_extractor.chunk_length = (
            max_rounded_up_audio_length_in_seconds
        )
        self.processor.feature_extractor.n_samples = (
            max_rounded_up_audio_length_in_seconds * sr
        )
        self.processor.feature_extractor.nb_max_frames = math.ceil(
            self.processor.feature_extractor.n_samples
            / self.processor.feature_extractor.hop_length
        )
        # Get the Mel spectrograms
        processed_audios = super()._process_audios(audios, sr, **kwargs)
        # Restore the original maximum audio length before returning
        self.processor.feature_extractor.chunk_length = old_chunk_length
        self.processor.feature_extractor.n_samples = old_n_samples
        self.processor.feature_extractor.nb_max_frames = old_nb_max_frames
        return processed_audios

    def _encode_processed_audios(
        self, input_values, attention_mask: Optional[torch.BoolTensor] = None
    ) -> SpeechLmmModuleOutput:
        # Trick Whisper into thinking that the maximum sequence length
        # is the maximum length of the encoded audios in this batch
        SEQ_LEN_DIM = 2
        input_length_after_conv1 = compute_output_length_from_conv1d_layer(
            input_values.shape[SEQ_LEN_DIM], conv1d_layer=self.model.conv1
        )
        input_length_after_conv2 = compute_output_length_from_conv1d_layer(
            input_length_after_conv1, conv1d_layer=self.model.conv2
        )
        old_max_source_positions = self.model.config.max_source_positions
        old_embed_positions_weight = self.model.embed_positions.weight.data
        self.model.config.max_source_positions = input_length_after_conv2
        self.model.embed_positions.weight.data = (
            self.model.embed_positions.weight[
                : self.model.config.max_source_positions
            ]
        )
        # Perform the forward pass
        model_output = super()._encode_processed_audios(
            input_values, attention_mask
        )
        # Restore the original maximum sequence length before returning
        self.model.config.max_source_positions = old_max_source_positions
        self.model.embed_positions.weight.data = old_embed_positions_weight
        return model_output


class EncodecEncoderModel(EncodecModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        # Delete decoder-specific modules so that the corresponding keys
        # in the state_dict are ignored when doing `from_pretrained`
        del self.decoder

    def get_decoder(self):
        raise NotImplementedError("This model is an encoder-only model.")

    def _decode_frame(
        self, codes: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError("This model is an encoder-only model.")

    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        raise NotImplementedError("This model is an encoder-only model.")

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_scales: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecEncoderOutput]:
        if audio_scales is not None or audio_codes is not None:
            raise ValueError(
                "This is an encoder-only model, so you MUSTN'T provide "
                "either of `audio_scales` and `audio_codes`."
            )

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()
        return_dict = return_dict or self.config.return_dict
        return self.encode(
            input_values=input_values,
            padding_mask=padding_mask,
            bandwidth=bandwidth,
            return_dict=return_dict,
        )


class HfCodecEncoder(HfAudioEncoder):
    @torch.no_grad()
    def forward(
        self,
        audios_srs: List[Tuple[torch.FloatTensor, int]],
        perturb_prob: float = 0.0,
    ) -> SpeechLmmModuleOutput:
        audios, sampling_rates = zip(*audios_srs)
        unique_sampling_rates = set(sampling_rates)
        if len(unique_sampling_rates) > 1:
            raise ValueError(
                "All audios must have the same sampling rate. "
                f"Found {len(unique_sampling_rates)} unique sampling rates: "
                f"{unique_sampling_rates}."
            )

        audios = [a.squeeze().float().cpu().numpy() for a in audios]
        input_values, attention_mask = self._process_audios(
            audios, sr=unique_sampling_rates.pop()
        )
        return self._encode_processed_audios(
            input_values, attention_mask, perturb_prob
        )


class EncodecEncoder(HfCodecEncoder):
    processor_class = AutoFeatureExtractor
    processor_audio_arg_name = "raw_audio"
    model_class = EncodecEncoderModel
    codebook2bandwidth = {2: 1.5, 4: 3.0, 8: 6.0, 16: 12.0, 32: 24.0}

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        delay_load: bool = False,
        bandwidth: Optional[float] = None,
    ):
        super().__init__(
            name_or_path=name_or_path,
            config_dict=config_dict,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            delay_load=delay_load,
        )

        num_quantizers = config_dict.get("n_quantizers")
        print(
            f"{self.__class__.__name__} initialized with num_quantizers_in_use={num_quantizers}"
        )
        self.perturb_codes = config_dict.get("perturb_codes", False)
        if self.perturb_codes:
            self.storm = CodeStorm(
                codebook_size=self.config.codebook_size,
                num_quantizers=num_quantizers,
            )
        self.bandwidth = (
            bandwidth
            if bandwidth is not None
            else self.codebook2bandwidth.get(num_quantizers)
        )
        # ↑ if None, the model will use the smallest supported bandwidth

    def _process_audios(self, audios: List[np.ndarray], sr: int, **kwargs):
        input_values, attention_mask = super()._process_audios(
            audios, sr, padding=True, **kwargs
        )
        # NOTE: for some weird reason, even though EnCodec's `forward`
        # method expects both `input_values` and `padding_mask` to be of
        # shape (batch_size, num_channels, num_samples), the processor
        # returns `input_values` of shape
        # (batch_size, num_channels, num_samples) and a `padding_mask`
        # of shape (batch_size, num_samples). Therefore, we need to
        # unsqueeze the `padding_mask` along the channel dimension
        # before returning it
        return input_values, attention_mask.unsqueeze(1)

    def _encode_processed_audios(
        self,
        input_values,
        attention_mask,
        perturb_prob: float = 0.0,
    ) -> CodecOutput:
        model_output = self.model(
            input_values,
            padding_mask=attention_mask,  # this is NOT a mistake
            bandwidth=self.bandwidth,
            return_dict=True,
            **self.model_forward_kwargs,
        )
        concatenated_audio_codes = torch.cat(
            torch.unbind(model_output.audio_codes, dim=0), dim=-1
        )
        audio_features = self.get_features_from_codes(concatenated_audio_codes)

        attention_mask = self._manually_recompute_attention_mask(
            attention_mask.squeeze(1)
            # ↑ must squeeze the dummy dimension we added in the
            # `_process_audios` method
        )
        encoded_frames = model_output.audio_codes
        codes = torch.cat(
            [encoded for encoded in encoded_frames], dim=-1
        )  # [batch, num_quantizers, timesteps]
        codes = einops.rearrange(
            codes, "b q n -> b n q"
        )  # result: [batch, timesteps, num_quantizers]
        if self.perturb_codes:
            corrupted = self.storm(codes=codes, perturb_prob=perturb_prob)
            audio_features = self.get_features_from_codes(corrupted)
        else:
            audio_features = self.get_features_from_codes(codes)
        return CodecOutput(
            features=audio_features,
            padding_mask=~attention_mask,
            audio_features_per_codebook=None,
            codes=codes,
        )

    def supports_output_attentions(self) -> bool:
        return False  # EnCodec doesn't even have attention layers

    def _manually_recompute_attention_mask(
        self, attention_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        SEQ_LEN_DIM = 1
        input_lengths = attention_mask.sum(dim=SEQ_LEN_DIM)
        rescaling_factor = self.output_sampling_rate / self.input_sampling_rate
        output_lengths = (input_lengths * rescaling_factor).ceil()
        return lengths_to_attention_mask(output_lengths.long())

    def get_features_from_codes(
        self, audio_codes: torch.Tensor
    ) -> torch.Tensor:
        permuted_audio_codes = audio_codes.permute(1, 0, 2)
        # ↑ (batch_size, n_quant, n_frames) -> (n_quant, batch_size, n_frames)
        audio_features = self.model.quantizer.decode(permuted_audio_codes)
        return audio_features.permute(0, 2, 1)
        # ↑ (batch_size, emb_dim, seq_len) -> (batch_size, seq_len, emb_dim)

    @property
    def output_sampling_rate(self) -> float:
        return float(self.config.frame_rate)


class MimiEncoderModel(MimiModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        # Delete decoder-specific modules so that the corresponding keys
        # in the state_dict are ignored when doing `from_pretrained`
        del self.decoder
        del self.decoder_transformer
        del self.upsample

    def get_decoder(self):
        raise NotImplementedError("This model is an encoder-only model.")

    def _decode_frame(
        self,
        codes: torch.Tensor,
        past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("This model is an encoder-only model.")

    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiDecoderOutput]:
        raise NotImplementedError("This model is an encoder-only model.")

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        decoder_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiEncoderOutput]:
        if audio_codes is not None:
            raise ValueError(
                "This is an encoder-only model, so you MUSTN'T provide"
                "`audio_codes`."
            )

        padding_mask = padding_mask or torch.ones_like(input_values).bool()
        return_dict = return_dict or self.config.return_dict

        return self.encode(
            input_values=input_values,
            padding_mask=padding_mask,
            num_quantizers=(
                num_quantizers
                if num_quantizers is not None
                else self.config.n_quantizers
            ),
            encoder_past_key_values=encoder_past_key_values,
            return_dict=return_dict,
        )


class MimiEncoder(HfCodecEncoder):
    model_class = MimiEncoderModel

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        delay_load: bool = False,
    ):
        super().__init__(
            name_or_path=name_or_path,
            config_dict=config_dict,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            delay_load=delay_load,
        )
        self.num_quantizers_in_use = config_dict.get("n_quantizers", 8)
        print(
            f"{self.__class__.__name__} initialized with num_quantizers_in_use={self.num_quantizers_in_use}"
        )
        self.perturb_codes = config_dict.get("perturb_codes", False)
        if self.perturb_codes:
            self.storm = CodeStorm(
                codebook_size=self.config.codebook_size,
                num_quantizers=self.num_quantizers_in_use,
            )

    def _encode_processed_audios(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
        perturb_prob: float = 0.0,
    ):
        padding_mask = attention_mask  # this is NOT a mistake
        # Extract discrete codes from Mimi
        with torch.inference_mode():
            self.model.eval()
            assert self.model.training == False, "Model must be in eval mode"
            output = self.model.encode(
                input_values=input_values,
                padding_mask=padding_mask,
                num_quantizers=self.num_quantizers_in_use,
            )
        # extract codes
        codes = output.audio_codes
        codes = einops.rearrange(
            codes, "b q n -> b n q"
        )  # result: [batch, timesteps, num_quantizers]
        if self.perturb_codes:
            corrupted = self.storm(codes=codes, perturb_prob=perturb_prob)
            audio_features = self.get_features_from_codes(corrupted)
        else:
            audio_features = self.get_features_from_codes(codes)

        assert (
            len(audio_features.size()) == 3
        )  # (batch_size, seq_len, hidden_size)
        attention_mask = (
            torch.nn.functional.interpolate(
                attention_mask.unsqueeze(0)
                .unsqueeze(0)
                .to(audio_features.dtype),
                size=(audio_features.shape[:2]),
                mode="bicubic",
                align_corners=False,
            )
            .to(dtype=torch.bool)
            .squeeze(0)
            .squeeze(0)
        )  # 12.5Hz

        return CodecOutput(
            features=audio_features,
            padding_mask=~attention_mask,
            audio_features_per_codebook=None,
            codes=codes,
        )

    def get_features_from_codes(
        self, audio_codes: torch.Tensor
    ) -> torch.Tensor:
        permuted_audio_codes = einops.rearrange(audio_codes, "b n q -> b q n")
        # ↑ (batch_size, n_quant, n_frames) -> (n_quant, batch_size, n_frames)
        audio_features = self.model.quantizer.decode(permuted_audio_codes)
        return einops.rearrange(audio_features, "b c n -> b n c")
        # ↑ (batch_size, emb_dim, seq_len) -> (batch_size, seq_len, emb_dim)

    @property
    def output_sampling_rate(self) -> float:
        return float(self.config.frame_rate)


AutoConfig.register("whisper-unrestricted", UnrestrictedWhisperConfig)
