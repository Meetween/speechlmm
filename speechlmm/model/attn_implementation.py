import logging
from typing import Optional


class AttentionImplementationMixin:
    def __init__(self):
        super().__init__()
        required_attrs = ["_supports_flash_attn_2", "_supports_sdpa"]
        if not all(hasattr(self, attr) for attr in required_attrs):
            raise ValueError(
                f"{AttentionImplementationMixin.__name__} must be used in "
                f"conjunction with a class having the following attributes: "
                f"{required_attrs}"
            )

    def set_attn_implementation_with_fallback(
        self, attn_implementation: Optional[str] = None
    ):
        supported_attn_implementations = ["eager", "sdpa", "flash_attention_2"]
        if (
            attn_implementation is not None
            and attn_implementation not in supported_attn_implementations
        ):
            raise ValueError(
                f"Attention implementation '{attn_implementation}' is not "
                f"supported. Supported implementations are "
                f"{supported_attn_implementations}."
            )

        if (
            attn_implementation == "flash_attention_2"
            and not self._supports_flash_attn_2
        ):
            logging.warning(
                f"{self.__class__.__qualname__} does not support "
                f"attn_implementation='flash_attention_2'. Trying again with "
                f"attn_implementation='sdpa'."
            )
            attn_implementation = "sdpa"

        if attn_implementation == "sdpa" and not self._supports_sdpa:
            logging.warning(
                f"{self.__class__.__qualname__} does not support "
                f"attn_implementation='sdpa'. Falling back to 'eager'."
            )
            attn_implementation = "eager"
            # NOTE: there exist models that do not support 'sdpa' but do
            # support 'flash_attention_2'. We do not try to fallback to
            # 'flash_attention_2' in this case because we would expect
            # the user to request 'flash_attention_2' directly

        self.attn_implementation = attn_implementation
        # NOTE: If `attn_implementation` is None, we just leave it like
        # that and let Hugging Face transformers set it to the default
        # value specified in the model configuration
