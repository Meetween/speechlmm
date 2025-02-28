from typing import Any, Dict, Optional

from transformers import AutoConfig, PretrainedConfig


class AdapterConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim


# Length adapter configs
class CifAdapterConfig(AdapterConfig):
    model_type = "cif"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        firing_threshold: float = 1.0,
        residual_threshold: float = 0.5,
        unbound_input_weights: bool = False,
        eps: float = 1e-4,
        quantity_loss_weight: float = 1.0,
        ctc_loss_weight: float = 0.0,
        ctc_loss_vocab_size: Optional[int] = None,
        ctc_loss_blank_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.cif = {
            "firing_threshold": firing_threshold,
            "residual_threshold": residual_threshold,
            "unbound_input_weights": unbound_input_weights,
            "eps": eps,
        }
        self.quantity_loss_weight = quantity_loss_weight
        self.ctc_loss_weight = ctc_loss_weight
        if self.ctc_loss_weight > 0:
            if ctc_loss_vocab_size is None or ctc_loss_blank_id is None:
                raise ValueError(
                    "CTC loss requires vocab size and blank id to be set."
                )
            self.ctc_loss_vocab_size = ctc_loss_vocab_size
            self.ctc_loss_blank_id = ctc_loss_blank_id


class CtcAdapterConfig(AdapterConfig):
    model_type = "ctc"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        ctc_loss_vocab_size: Optional[int] = None,
        ctc_loss_blank_id: Optional[int] = None,
        ctc_loss_weight: float = 1.0,
        compression_strategy: str = "mean",
        max_compressed_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.ctc_loss_vocab_size = ctc_loss_vocab_size
        self.ctc_loss_blank_id = ctc_loss_blank_id
        if ctc_loss_weight <= 0:
            raise ValueError("CTC loss weight must be positive.")
        self.ctc_loss_weight = ctc_loss_weight
        self.compression_strategy = compression_strategy
        self.max_compressed_length = max_compressed_length


class ConvolutionalAdapterConfig(AdapterConfig):
    model_type = "conv"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_layers: int = 0,
        hidden_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size or self.input_dim


# Modality adapter configs
class MlpAdapterConfig(AdapterConfig):
    model_type = "mlp"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_layers: int = 2,
        hidden_size: int = 4096,
        residual_type: str = "interpolation",
        force_input_projection: bool = True,
        force_output_projection: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.residual_type = residual_type
        self.force_input_projection = force_input_projection
        self.force_output_projection = force_output_projection


class TransformerAdapterConfig(AdapterConfig):
    model_type = "transformer"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


# Full adapter configs
class TwoStageAdapterConfig(AdapterConfig):
    is_composition = True
    modality_adapter_type = None

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        length_adapter: Optional[Dict[str, Any]] = None,
        modality_adapter_before: Optional[Dict[str, Any]] = None,
        modality_adapter_after: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        if length_adapter is None:
            raise ValueError("Length adapter is required.")
        self.length_adapter = AutoConfig.for_model(**length_adapter)

        if modality_adapter_before is None and modality_adapter_after is None:
            raise ValueError(
                "At least one modality adapter (before, after) is required."
            )

        if modality_adapter_before is not None:
            if (
                self.modality_adapter_type is not None
                and modality_adapter_before["model_type"]
                != self.modality_adapter_type
            ):
                raise ValueError(
                    f"Modality adapter (before) must be of type "
                    f"'{self.modality_adapter_type}', not "
                    f"'{modality_adapter_before['model_type']}'."
                )
            self.modality_adapter_before = AutoConfig.for_model(
                **modality_adapter_before
            )

        if modality_adapter_after is not None:
            if (
                self.modality_adapter_type is not None
                and modality_adapter_after["model_type"]
                != self.modality_adapter_type
            ):
                raise ValueError(
                    f"Modality adapter (after) must be of type "
                    f"'{self.modality_adapter_type}', not "
                    f"'{modality_adapter_after['model_type']}'."
                )
            self.modality_adapter_after = AutoConfig.for_model(
                **modality_adapter_after
            )

    @classmethod
    def from_length_modality_adapter_configs(
        cls,
        length_adapter_config: AdapterConfig,
        modality_adapter_before_config: Optional[AdapterConfig] = None,
        modality_adapter_after_config: Optional[AdapterConfig] = None,
    ):
        if (
            modality_adapter_before_config is None
            and modality_adapter_after_config is None
        ):
            raise ValueError(
                "At least one modality adapter (before, after) is required."
            )

        input_dim = length_adapter_config.input_dim
        output_dim = length_adapter_config.output_dim

        modality_adapter_before_config_dict = None
        if modality_adapter_before_config is not None:
            modality_adapter_before_config_dict = (
                modality_adapter_before_config.to_dict()
            )
            input_dim = modality_adapter_before_config.input_dim

        modality_adapter_after_config_dict = None
        if modality_adapter_after_config is not None:
            modality_adapter_after_config_dict = (
                modality_adapter_after_config.to_dict()
            )
            output_dim = modality_adapter_after_config.output_dim

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            length_adapter=length_adapter_config.to_dict(),
            modality_adapter_before=modality_adapter_before_config_dict,
            modality_adapter_after=modality_adapter_after_config_dict,
        )

    def to_diff_dict(self) -> Dict[str, Any]:
        diff_dict = super().to_diff_dict()
        diff_dict["length_adapter"] = self.length_adapter.to_diff_dict()
        if hasattr(self, "modality_adapter_before"):
            diff_dict["modality_adapter_before"] = (
                self.modality_adapter_before.to_diff_dict()
            )
        if hasattr(self, "modality_adapter_after"):
            diff_dict["modality_adapter_after"] = (
                self.modality_adapter_after.to_diff_dict()
            )

        return diff_dict


class CformerAdapterConfig(TwoStageAdapterConfig):
    model_type = "cformer"
    is_composition = True
    modality_adapter_type = "transformer"


class CmlpAdapterConfig(TwoStageAdapterConfig):
    model_type = "cmlp"
    is_composition = True
    modality_adapter_type = "mlp"


class WindowLevelQformerAdapterConfig(AdapterConfig):
    model_type = "qformer"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        add_cross_attention: bool = True,
        num_queries: int = 1,
        cross_attention_every_n_layers: int = 1,
        window_size_in_seconds: float = 0.3333333333333,  # leaved for retrocompatibility
        compress_factor: Optional[int] = None,
        hop_size: float = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            add_cross_attention=add_cross_attention,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_queries = num_queries
        self.cross_attention_every_n_layers = cross_attention_every_n_layers
        self.window_size_in_seconds = window_size_in_seconds
        self.hop_size = hop_size
        self.compress_factor = compress_factor
        if kwargs.get("triplet_loss", None) is not None:
            self.triplet_loss = kwargs.get("triplet_loss")


class BackfeedingAdapterConfig(PretrainedConfig):
    def __init__(
        self,
        audio_adapter: AdapterConfig = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_adapter = audio_adapter


class BackfeedingAdapterOnFeaturesConfig(BackfeedingAdapterConfig):
    model_type = "features"


class BackfeedingAdapterOnCodesConfig(BackfeedingAdapterConfig):
    model_type = "codes"


AutoConfig.register(model_type="cif", config=CifAdapterConfig)
AutoConfig.register(model_type="ctc", config=CtcAdapterConfig)
AutoConfig.register(model_type="conv", config=ConvolutionalAdapterConfig)
AutoConfig.register(model_type="mlp", config=MlpAdapterConfig)
AutoConfig.register(model_type="transformer", config=TransformerAdapterConfig)
AutoConfig.register(model_type="cformer", config=CformerAdapterConfig)
AutoConfig.register(model_type="cmlp", config=CmlpAdapterConfig)
AutoConfig.register(
    model_type="qformer", config=WindowLevelQformerAdapterConfig
)
AutoConfig.register(
    model_type="features", config=BackfeedingAdapterOnFeaturesConfig
)
AutoConfig.register(model_type="codes", config=BackfeedingAdapterOnCodesConfig)
