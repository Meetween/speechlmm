import logging
from typing import Any, Dict, List, Optional

from transformers import AutoConfig, PretrainedConfig

from speechlmm.model.encoders.video_encoder import AutoAvsrConfig


class SpeechLmmConfig(PretrainedConfig):
    model_type = "speechlmm"
    is_composition = True

    def __init__(
        self,
        add_lm_head: bool = True,
        ################################################################
        # TODO(anferico):
        # - give more descriptive names to these arguments
        # - remove "mm_"
        # - use "vision_" and "audio_" (not "speech_") prefixes
        # - merge when possible (e.g., `mm_use_im_start_end` and
        #   `mm_use_audio_start_end`)
        conversation_version: str = None,
        vision_select_layer: int = -1,
        vision_use_patch_token: bool = True,
        vision_patch_merge_type: str = "flat",
        vision_select_feature: str = "patch",
        mm_use_im_start_end: bool = False,
        mm_use_audio_start_end: bool = False,
        mm_use_video_start_end: bool = False,
        tokenizer_padding_side: str = "right",
        # TODO(anferico): all these parameters below should go in the
        # appropriate subconfigs (e.g. audio_encoder, talking_head, ...)
        chunk_size_in_seconds: Optional[float] = None,
        chunk_overlap_in_seconds: float = 0.0,
        chunk_encoding_strategy: str = "loop",  # [batch, loop]
        perturb_codes: bool = False,
        perturb_prob: float = 0.0,
        use_audio_encoder_as_codec_encoder: bool = False,
        pad_audio_weight: float = 0.5,
        epad_audio_weight: float = 1,
        pad_epad_audio_weight_decay: float = 0.5,
        perturb_prob_decay: float = 0.5,
        codebook_weights: List[float] = None,
        audio_loss_decay: float = 0.5,
        audio_loss_weight: float = 0.1,
        ################################################################
        **kwargs,
    ):
        super().__init__(**kwargs)

        # NOTE: this weird pattern of popping arguments from `kwargs`
        # rather than putting them in the signature is taken from the
        # Hugging Face transformers codebase (see e.g.
        # `EncoderDecoderConfig` and `RagConfig`, which are composite
        # config classes (`is_composition = True`) like `SpeechLmmConfig`).
        # I'm not sure why they do it, but I'm going to assume there's
        # a good reason for it and keep it here.

        if "text_decoder" not in kwargs:
            raise ValueError("`text_decoder` must be provided.")
        text_decoder_config = kwargs.pop("text_decoder")
        text_decoder_model_type = text_decoder_config.pop("model_type")
        self.text_decoder = AutoConfig.for_model(
            text_decoder_model_type, **text_decoder_config
        )
        # TODO(anferico): added this to please DeepSpeed, but is this
        # the right way to do it? Cause technically a `SpeechLmmModel` has
        # multiple hidden sizes (encoder, adapter, decoder). I guess
        # what matters is the dimensionality of the *output* embeddings?
        self.hidden_size = self.text_decoder.hidden_size

        if "vision_encoder" in kwargs:
            if "vision_adapter" not in kwargs:
                raise ValueError(
                    "`vision_encoder` is provided, but `vision_adapter` is"
                    " missing."
                )
            vision_encoder_config = kwargs.pop("vision_encoder")
            vision_encoder_model_type = vision_encoder_config.pop("model_type")
            vision_adapter_config = kwargs.pop("vision_adapter")
            vision_adapter_model_type = vision_adapter_config.pop("model_type")
            self.vision_encoder = AutoConfig.for_model(
                vision_encoder_model_type, **vision_encoder_config
            )
            self.vision_adapter = AutoConfig.for_model(
                vision_adapter_model_type, **vision_adapter_config
            )

        if "audio_encoder" in kwargs:
            if "audio_adapter" not in kwargs:
                raise ValueError(
                    "`audio_encoder` is provided, but `audio_adapter` is"
                    " missing."
                )

            audio_encoder_config = kwargs.pop("audio_encoder")
            audio_encoder_model_type = audio_encoder_config.pop("model_type")
            audio_adapter_config = kwargs.pop("audio_adapter")
            audio_adapter_model_type = audio_adapter_config.pop("model_type")
            self.audio_encoder = AutoConfig.for_model(
                audio_encoder_model_type, **audio_encoder_config
            )
            self.audio_adapter = AutoConfig.for_model(
                audio_adapter_model_type, **audio_adapter_config
            )

        if "talking_head" in kwargs:
            if "backfeeding_audio_adapter" not in kwargs:
                raise ValueError(
                    "`talking_head` is provided, but `backfeeding_audio_adapter` is"
                    " missing."
                )

            if "codec_decoder" not in kwargs:
                raise ValueError(
                    "`talking_head` is provided, but `backfeeding_audio_adapter` is"
                    " missing."
                )

            if "codec_encoder" not in kwargs:
                raise ValueError(
                    "`talking_head` is provided, but `codec_encoder` is"
                    " missing."
                )

            codec_encoder_config = kwargs.pop("codec_encoder")
            codec_encoder_model_type = codec_encoder_config.pop("model_type")

            talking_head_config = kwargs.pop("talking_head")
            talking_head_model_type = talking_head_config.pop("model_type")

            self.talking_head = AutoConfig.for_model(
                talking_head_model_type, **talking_head_config
            )

            backfeeding_audio_adapter_config = kwargs.pop(
                "backfeeding_audio_adapter"
            )
            backfeeding_audio_adapter_model_type = (
                backfeeding_audio_adapter_config.pop("model_type")
            )
            self.backfeeding_audio_adapter = AutoConfig.for_model(
                backfeeding_audio_adapter_model_type,
                **backfeeding_audio_adapter_config,
            )

            codec_decoder_config = kwargs.pop("codec_decoder")
            codec_decoder_model_type = codec_decoder_config.pop("model_type")
            self.codec_decoder = AutoConfig.for_model(
                codec_decoder_model_type, **codec_decoder_config
            )

            self.codec_encoder = AutoConfig.for_model(
                codec_encoder_model_type, **codec_encoder_config
            )
            self.codec_encoder.perturb_codes = perturb_codes

            if "conditioning_audio_adapter" in kwargs:
                conditioning_audio_adapter_config = kwargs.pop(
                    "conditioning_audio_adapter"
                )
                conditioning_audio_adapter_model_type = (
                    conditioning_audio_adapter_config.pop("model_type")
                )
                self.conditioning_audio_adapter = AutoConfig.for_model(
                    conditioning_audio_adapter_model_type,
                    **conditioning_audio_adapter_config,
                )
            self.perturb_codes = perturb_codes
            self.perturb_prob = perturb_prob
            self.pad_audio_weight = pad_audio_weight
            self.epad_audio_weight = epad_audio_weight
            self.pad_epad_audio_weight_decay = pad_epad_audio_weight_decay
            self.perturb_prob_decay = perturb_prob_decay
            self.audio_loss_decay = audio_loss_decay
            self.audio_loss_weight = audio_loss_weight
            self.use_audio_encoder_as_codec_encoder = (
                use_audio_encoder_as_codec_encoder
            )
            self.codebook_weights = codebook_weights

        if "video_encoder" in kwargs:
            if "video_adapter" not in kwargs:
                raise ValueError(
                    "`video_encoder` is provided, but `video_adapter` is"
                    "missing."
                )

            video_encoder_config = kwargs.pop("video_encoder")
            self.video_encoder = AutoAvsrConfig(**video_encoder_config)
            video_adapter_config = kwargs.pop("video_adapter")
            video_adapter_model_type = video_adapter_config.pop("model_type")
            self.video_adapter = AutoConfig.for_model(
                video_adapter_model_type, **video_adapter_config
            )

            self.codebook_weights = codebook_weights

        self.add_lm_head = add_lm_head
        self.conversation_version = conversation_version
        self.vision_select_layer = vision_select_layer
        self.mm_use_im_start_end = mm_use_im_start_end
        self.vision_use_patch_token = vision_use_patch_token
        self.vision_patch_merge_type = vision_patch_merge_type
        self.vision_select_feature = vision_select_feature
        self.mm_use_audio_start_end = mm_use_audio_start_end
        self.mm_use_video_start_end = mm_use_video_start_end
        self.tokenizer_padding_side = tokenizer_padding_side
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.chunk_overlap_in_seconds = chunk_overlap_in_seconds
        self.chunk_encoding_strategy = chunk_encoding_strategy

    @classmethod
    def from_encoders_adapters_decoder_configs(
        cls,
        text_decoder: PretrainedConfig,
        vision_encoder: Optional[PretrainedConfig] = None,
        vision_adapter: Optional[PretrainedConfig] = None,
        audio_encoder: Optional[PretrainedConfig] = None,
        audio_adapter: Optional[PretrainedConfig] = None,
        talking_head: Optional[PretrainedConfig] = None,
        backfeeding_audio_adapter: Optional[PretrainedConfig] = None,
        codec_decoder: Optional[PretrainedConfig] = None,
        codec_encoder: Optional[PretrainedConfig] = None,
        conditioning_audio_adapter: Optional[PretrainedConfig] = None,
        video_encoder: Optional[PretrainedConfig] = None,
        video_adapter: Optional[PretrainedConfig] = None,
        **kwargs,
    ):
        if vision_encoder is None and audio_encoder is None:
            raise ValueError(
                "At least one of vision_encoder and audio_encoder must be "
                "provided."
            )

        multimodal_modules = dict()
        if vision_encoder is not None:
            if vision_adapter is None:
                raise ValueError(
                    "`vision_encoder` is provided, but `vision_adapter` is"
                    "missing."
                )
            multimodal_modules.update(
                {
                    "vision_encoder": vision_encoder.to_dict(),
                    "vision_adapter": vision_adapter.to_dict(),
                }
            )

        if audio_encoder is not None:
            if audio_adapter is None:
                raise ValueError(
                    "`audio_encoder` is provided, but `audio_adapter` is"
                    "missing."
                )
            multimodal_modules.update(
                {
                    "audio_encoder": audio_encoder.to_dict(),
                    "audio_adapter": audio_adapter.to_dict(),
                }
            )

        if video_encoder is not None:
            if video_adapter is None:
                raise ValueError(
                    "`video_encoder` is provided, but `video_adapter` is"
                    "missing."
                )
            multimodal_modules.update(
                {
                    "video_encoder": video_encoder.to_dict(),
                    "video_adapter": video_adapter.to_dict(),
                }
            )

        if talking_head is not None:
            if backfeeding_audio_adapter is None:
                raise ValueError(
                    "`talking_head` is provided, but `backfeeding_audio_adapter` is"
                    "missing."
                )
            if codec_decoder is None:
                raise ValueError(
                    "`talking_head` is provided, but `codec_decoder` is"
                    "missing."
                )
            if codec_encoder is None:
                raise ValueError(
                    "`talking_head` is provided, but `codec_encoder` is"
                    "missing."
                )

            multimodal_modules.update(
                {
                    "talking_head": talking_head.to_dict(),
                    "backfeeding_audio_adapter": backfeeding_audio_adapter.to_dict(),
                    "codec_decoder": codec_decoder.to_dict(),
                    "codec_encoder": codec_encoder.to_dict(),
                }
            )
            if conditioning_audio_adapter is not None:
                multimodal_modules.update(
                    {
                        "conditioning_audio_adapter": conditioning_audio_adapter.to_dict(),
                    }
                )

        return cls(
            text_decoder=text_decoder.to_dict(),
            **multimodal_modules,
            **kwargs,
        )

    def to_diff_dict(self) -> Dict[str, Any]:
        diff_dict = super().to_diff_dict()
        diff_dict["text_decoder"] = self.text_decoder.to_diff_dict()
        if hasattr(self, "vision_encoder"):
            diff_dict["vision_encoder"] = self.vision_encoder.to_diff_dict()
            diff_dict["vision_adapter"] = self.vision_adapter.to_diff_dict()
        if hasattr(self, "audio_encoder"):
            diff_dict["audio_encoder"] = self.audio_encoder.to_diff_dict()
            diff_dict["audio_adapter"] = self.audio_adapter.to_diff_dict()
        if hasattr(self, "talking_head"):
            diff_dict["talking_head"] = self.talking_head.to_diff_dict()
            diff_dict["backfeeding_audio_adapter"] = (
                self.backfeeding_audio_adapter.to_diff_dict()
            )
            diff_dict["codec_decoder"] = self.codec_decoder.to_diff_dict()
            diff_dict["codec_encoder"] = self.codec_encoder.to_diff_dict()
            if hasattr(self, "conditioning_audio_adapter"):
                diff_dict["conditioning_audio_adapter"] = (
                    self.conditioning_audio_adapter.to_diff_dict()
                )

        if hasattr(self, "video_encoder"):
            diff_dict["video_encoder"] = self.video_encoder.to_diff_dict()
            diff_dict["video_adapter"] = self.video_adapter.to_diff_dict()

        return diff_dict
