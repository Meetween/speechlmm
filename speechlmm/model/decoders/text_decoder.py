import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
    MistralForCausalLM,
    MistralModel,
    PreTrainedModel,
)

from speechlmm import conversation as conversation_lib
from speechlmm.model.attn_implementation import AttentionImplementationMixin

logger = logging.getLogger(__name__)


class HfTextDecoder(
    AttentionImplementationMixin, torch.nn.Module, metaclass=ABCMeta
):
    """
    Base class for Hugging Face text decoders. This class is a wrapper
    around Hugging Face text decoders that provides a consistent interface for
    decoding multimodal embeddings into text.

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
    tokenizer_class = AutoTokenizer
    tokenizer_text_arg_name = None
    model_class = None
    model_for_causal_lm_class = None
    model_forward_kwargs = {}

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        add_lm_head: bool = True,
        tokenizer_padding_side: str = "right",
        conversation_version: Optional[str] = None,
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

        if add_lm_head and (
            self.model_for_causal_lm_class is None
            or not issubclass(self.model_for_causal_lm_class, PreTrainedModel)
        ):
            raise ValueError(
                f"Class attribute `model_for_causal_lm_class` must be a "
                f"subclass of `transformers.PreTrainedModel` (found "
                f"{self.model_for_causal_lm_class})."
            )
        self.add_lm_head = add_lm_head

        config_dict = config_dict or {}
        name_or_path = name_or_path or config_dict.pop("_name_or_path", None)
        if name_or_path is None:
            raise ValueError(
                "`name_or_path` must be provided either as an explicit "
                "argument or as part of `config_kwargs` (in which case it "
                "should be named `_name_or_path`)."
            )
        self.name_or_path = name_or_path

        torch_dtype_in_config = config_dict.pop("torch_dtype", None)
        torch_dtype = torch_dtype or torch_dtype_in_config
        self.config = self.config_class.from_pretrained(
            self.name_or_path, torch_dtype=torch_dtype, **config_dict
        )

        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.name_or_path,
            cache_dir=cache_dir,
            padding_side=tokenizer_padding_side,
        )

        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self._load_model()

        # NOTE: _default_conversation_version is deprecated
        # please set conversation_version in text decoder config
        # conversation_version = (
        #     conversation_version or self._default_conversation_version
        # )
        if conversation_version is None:
            raise ValueError("`conversation_version` must be provided.")
        self.conversation_version = conversation_version
        self._adapt_tokenizer_to_conversation_version()

    @property
    @abstractmethod
    def _default_conversation_version(self):
        pass

    def _adapt_tokenizer_to_conversation_version(self):
        if self.conversation_version == "v0":
            if self.tokenizer.pad_token is None:
                pad_token = "[PAD]"
                logger.info(f"Adding pad token '{pad_token}'.")
                self._resize_tokenizer_and_embedding_layer(
                    additional_special_tokens=[pad_token],
                )
        elif self.conversation_version == "v0.5":
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.conversation_version == "v1":
            self.tokenizer.pad_token = self.tokenizer.unk_token
            if self.conversation_version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = (
                    conversation_lib.conv_templates[self.conversation_version]
                )
            else:
                conversation_lib.default_conversation = (
                    conversation_lib.conv_templates["vicuna_v1"]
                )
        elif self.conversation_version == "mistral_instruct":
            self.tokenizer.pad_token = self.tokenizer.unk_token
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates["mistral_instruct"]
            )
        elif (
            self.conversation_version == "llama_3_1"
            or self.conversation_version == "llama_3_1_base"
        ):
            pad_token = "<|finetune_right_pad_id|>"
            logger.info(f"Setting pad token to '{pad_token}'.")
            self.tokenizer.pad_token = pad_token
            self.tokenizer.pad_token_id = 128004
            self.model.generation_config.eos_token_id = 128009
            # ↑ 128009 = <|eot_id|>
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates["llama_3_1"]
                if self.conversation_version == "llama_3_1"
                else conversation_lib.conv_templates["llama_3_1_base"]
            )
        else:
            if (
                self.conversation_version
                not in conversation_lib.conv_templates
            ):
                raise ValueError(
                    f"Unknown conversation format "
                    f"'{self.conversation_version}'."
                )
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates[self.conversation_version]
            )
            if self.tokenizer.pad_token is None:
                pad_token = "<pad>"
                logger.info(f"Adding pad token as {pad_token}")
                self._resize_tokenizer_and_embedding_layer(
                    additional_special_tokens=[pad_token],
                )

        logger.info(
            f"Using conversation format: "
            f"{conversation_lib.default_conversation.version}"
        )

    def _resize_tokenizer_and_embedding_layer(
        self,
        additional_special_tokens: List[str],
    ):
        """
        Note: This is the unoptimized version that may make your embedding size
        not be divisible by 64.
        """
        special_tokens_dict = {
            "additional_special_tokens": additional_special_tokens
        }
        num_new_tokens = self.tokenizer.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens=False
        )
        if num_new_tokens > 0:  # i.e. if they were not already there
            self.model.resize_token_embeddings(
                new_num_tokens=len(self.tokenizer)
            )
            input_embeddings = self.model.get_input_embeddings().weight.data
            mean_input_embedding = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_new_tokens:] = mean_input_embedding

            output_embeddings = self.model.get_output_embeddings().weight.data
            mean_output_embedding = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings[-num_new_tokens:] = mean_output_embedding

    def _load_model(self):
        target_class = self.model_class
        if self.add_lm_head:
            target_class = self.model_for_causal_lm_class

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.torch_dtype or default_dtype)
        self.model = target_class.from_pretrained(
            self.name_or_path,
            config=self.config,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
        )
        torch.set_default_dtype(default_dtype)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Hook this wrapper's config to the model's config so that each
        # update to `self.config` is reflected in the model's config
        self.config = self.model.config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    # TODO(anferico): add method to preprocess text inputs?

    # TODO(anferico): add an `embed_tokens` method and replace calls to
    # `text_decoder.model.embed_tokens` with `text_decoder.embed_tokens`

    # TODO(anferico): this method should be more sophisticated than
    # this. In particular, if possible, it should implement the Adapter
    # pattern from OOP to adapt a common interface (HfTextDecoder) to
    # model-specific interfaces (e.g. MistralModel)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # TODO(anferico): this method should be more sophisticated than
    # this. In particular, if possible, it should implement the Adapter
    # pattern from OOP to adapt a common interface (HfTextDecoder) to
    # model-specific interfaces (e.g. MistralModel)
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


class MistralDecoder(HfTextDecoder):
    tokenizer_class = LlamaTokenizer
    tokenizer_text_arg_name = "text"
    model_class = MistralModel
    model_for_causal_lm_class = MistralForCausalLM

    @property
    def _default_conversation_version(self):
        return "mistral_instruct"


class LlamaDecoder(HfTextDecoder):
    tokenizer_class = AutoTokenizer
    tokenizer_text_arg_name = "text"
    model_class = LlamaModel
    model_for_causal_lm_class = LlamaForCausalLM

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        add_lm_head: bool = True,
        tokenizer_padding_side: str = "right",
        conversation_version: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            name_or_path=name_or_path,
            config_dict=config_dict,
            add_lm_head=add_lm_head,
            tokenizer_padding_side=tokenizer_padding_side,
            conversation_version=conversation_version,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        if "Llama-3" in self.model.config._name_or_path:
            if not "Llama-3." in self.model.config._name_or_path:
                raise ValueError("Only Llama 3.X models are supported.")

            # https://llama.com/docs/model-cards-and-prompt-formats/llama3_1
            pad_token, pad_token_id = "<|finetune_right_pad_id|>", 128004
            self.tokenizer.pad_token = pad_token
            self.tokenizer.pad_token_id = pad_token_id
            self.config.pad_token_id = pad_token_id

    @property
    def _default_conversation_version(self):
        if "vicuna" in self.model.config._name_or_path:
            return "v1"
        elif "Llama-3.1" in self.model.config._name_or_path:
            if "Instruct" in self.model.config._name_or_path:
                return "llama_3_1"
            else:
                return "llama_3_1_base"
        else:
            raise ValueError(
                f"Could not infer conversation version for model "
                f"'{self.model.config.model_name_or_path}'."
            )
