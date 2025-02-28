import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    EncodecConfig,
    EncodecModel,
    MimiConfig,
    MimiModel,
    PreTrainedModel,
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

from speechlmm.model.adapters.utils import lengths_to_attention_mask
from speechlmm.model.attn_implementation import AttentionImplementationMixin


class HfAudioDecoder(
    AttentionImplementationMixin, torch.nn.Module, metaclass=ABCMeta
):
    """
    Base class for Hugging Face audio decoders. This class is a wrapper
    around Hugging Face audio decoders that provides a consistent interface for
    decoding multimodal embeddings into audio.

    Args:
        name_or_path (str, optional): The model name or path to load the model
          from. If not provided, it will be inferred from `config_kwargs`.
        config_dict: Keyword arguments used to override the default values
          in the model configuration. Useful for loading pre-trained models
          with a configuration file attached to them.
        attn_implementation (str, optional): The attention implementation to
          use. If None, the default attention implementation from the model
          configuration is used. Defaults to None.
        torch_dtype (torch.dtype, optional): The data type to use for the
          model. If None, the default data type from the model configuration
          is used. Defaults to None.
        cache_dir (str, optional): The directory to cache pretrained weights.
    """

    config_class = AutoConfig
    model_class = None
    model_forward_kwargs = {}

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
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

        # NOTE: we call `super().__init__` now because
        # `AttentionImplementationMixin` expects to find the
        # `_supports_flash_attn_2` and `_supports_sdpa` attributes
        super().__init__()

        self.set_attn_implementation_with_fallback(attn_implementation)

        config_dict = config_dict or {}
        name_or_path = name_or_path or config_dict.pop("_name_or_path", None)
        if name_or_path is None:
            raise ValueError(
                "`name_or_path` must be provided either as an explicit "
                "argument or as part of `config_kwargs` (in which case it "
                "should be named `_name_or_path`)."
            )
        self.name_or_path = name_or_path

        self.config = self.config_class.from_pretrained(
            self.name_or_path, **config_dict
        )

        self.config_dict = config_dict

        self.model = self.model_class.from_pretrained(
            self.name_or_path,
            config=self.config,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir

    # TODO(anferico): this method should be more sophisticated than
    # this. In particular, if possible, it should implement the Adapter
    # pattern from OOP to adapt a common interface (HfAudioDecoder) to
    # model-specific interfaces (e.g. MimiModel)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    @abstractmethod
    def input_sampling_rate(self):
        pass

    # TODO(anferico): for now, we are assuming that audio decoders can
    # only be RVQ-based models
    @property
    @abstractmethod
    def num_quantizers(self):
        pass

    # TODO(anferico): for now, we are assuming that audio decoders can
    # only be RVQ-based models
    @property
    @abstractmethod
    def codebook_size(self):
        pass

    @property
    def hidden_size(self):
        return self.config.hidden_size


class MimiDecoderModel(MimiModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        # Delete encoder-specific modules so that the corresponding keys
        # in the state_dict are ignored when doing `from_pretrained`
        del self.encoder
        del self.encoder_transformer
        del self.downsample

    def get_encoder(self):
        raise NotImplementedError("This model is a decoder-only model.")

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        num_quantizers: int,
        padding_mask: int,
        past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("This model is a decoder-only model.")

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        num_quantizers: Optional[float] = None,
        encoder_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], MimiEncoderOutput]:
        raise NotImplementedError("This model is a decoder-only model.")

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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiDecoderOutput]:
        # NOTE: in `MimiModel.forward`, the `encode` method is only
        # called if `audio_codes` is None. By requiring `audio_codes` to
        # be not None, we ensure that the `encode` method is never
        # called
        if audio_codes is None:
            raise ValueError(
                "This is a decoder-only model, so you MUST provide `audio_codes`."
            )

        return super().forward(
            input_values=input_values,
            padding_mask=padding_mask,
            num_quantizers=num_quantizers,
            audio_codes=audio_codes,
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
            return_dict=return_dict,
        )


class MimiDecoder(HfAudioDecoder):
    config_class = MimiConfig
    model_class = MimiDecoderModel
    model_forward_kwargs = {}

    def forward(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # TODO(anferico): most of this method's code is copied from
        # `EncodecDecoder.forward`. Might want to refactor at some point

        if padding_mask is None:
            batch_size, _, num_codes = audio_codes.shape
            padding_mask = torch.zeros((batch_size, num_codes)).bool()
            # NOTE: yes, it's called padding mask but it's actually an
            # attention mask 😅

        # the padding mask passed as argument here is relative to the
        # audio codes, while the model's `forward` expects a padding
        # mask relative to the input values, so we must convert the
        # former to the latter
        # TODO(anferico): make sure this is correct
        audio_codes_lengths = (~padding_mask).sum(dim=-1)
        audio_lengths_in_seconds = (
            audio_codes_lengths / self.input_sampling_rate
        )
        audio_lengths_in_samples = math.ceil(
            audio_lengths_in_seconds * self.output_sampling_rate
        )
        inputs_padding_mask = lengths_to_attention_mask(
            audio_lengths_in_samples
        )

        decoder_output = self.model(
            input_values=None,
            padding_mask=inputs_padding_mask,
            num_quantizers=None,
            audio_codes=audio_codes,
            encoder_past_key_values=None,
            decoder_past_key_values=None,
            return_dict=True,
            **self.model_forward_kwargs,
        )
        return decoder_output.audio_values

    @property
    def input_sampling_rate(self):
        return self.config.frame_rate

    @property
    def output_sampling_rate(self):
        return self.config.sampling_rate

    @property
    def num_quantizers(self):
        return (
            self.config.num_quantizers
            if self.config_dict.get("n_quantizers", None) is None
            else self.config_dict["n_quantizers"]
        )

    @property
    def codebook_size(self):
        return self.config.codebook_size


class EncodecDecoderModel(EncodecModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        # Delete encoder-specific modules so that the corresponding keys
        # in the state_dict are ignored when doing `from_pretrained`
        del self.encoder

    def get_encoder(self):
        raise NotImplementedError("This model is a decoder-only model.")

    def _encode_frame(
        self, input_values: torch.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("This model is a decoder-only model.")

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], EncodecEncoderOutput
    ]:
        raise NotImplementedError("This model is a decoder-only model.")

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_scales: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        if audio_scales is None or audio_codes is None:
            # NOTE: in `EncodecModel.forward`, the `encode` method is
            # only called if `audio_scales` and `audio_codes` are both
            # None, and raises an error if only one of them is None.
            # By requiring both to be not None, we ensure that the
            # `encode` method is never called
            raise ValueError(
                "This is a decoder-only model, so you MUST provide "
                "`audio_scales` and `audio_codes`."
            )

        return super().forward(
            input_values=input_values,
            padding_mask=padding_mask,
            bandwidth=bandwidth,
            audio_codes=audio_codes,
            audio_scales=audio_scales,
            return_dict=return_dict,
        )


class EncodecDecoder(HfAudioDecoder):
    config_class = EncodecConfig
    model_class = EncodecDecoderModel
    model_forward_kwargs = {}

    def forward(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        audio_scales: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if padding_mask is None:
            batch_size, _, num_codes = audio_codes.shape
            padding_mask = torch.zeros((batch_size, num_codes)).bool()
            # NOTE: yes, it's called padding mask but it's actually an
            # attention mask 😅

        # the padding mask passed as argument here is relative to the
        # audio codes, while the model's `forward` expects a padding
        # mask relative to the input values, so we must convert the
        # former to the latter
        # TODO(anferico): make sure this is correct
        audio_codes_lengths = (~padding_mask).sum(dim=-1)
        audio_lengths_in_seconds = (
            audio_codes_lengths / self.input_sampling_rate
        )
        audio_lengths_in_samples = math.ceil(
            audio_lengths_in_seconds * self.output_sampling_rate
        )
        inputs_padding_mask = lengths_to_attention_mask(
            audio_lengths_in_samples
        )

        decoder_output = self.model(
            input_values=None,
            padding_mask=inputs_padding_mask,
            bandwidth=None,
            audio_codes=audio_codes,
            audio_scales=audio_scales,
            return_dict=True,
            **self.model_forward_kwargs,
        )
        return decoder_output.audio_values

    @property
    def input_sampling_rate(self):
        return self.config.frame_rate

    @property
    def output_sampling_rate(self):
        return self.config.sampling_rate

    @property
    def num_quantizers(self):
        return (
            self.config.num_quantizers
            if self.config_dict.get("n_quantizers", None) is None
            else self.config_dict["n_quantizers"]
        )

    @property
    def codebook_size(self):
        return self.config.codebook_size
