import contextlib
import itertools
import math
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoConfig, BertConfig, BertModel
from transformers.integrations.deepspeed import (
    deepspeed_config,
    is_deepspeed_zero3_enabled,
)

from speechlmm.constants import IGNORE_INDEX
from speechlmm.model.adapters.cif import Cif
from speechlmm.model.adapters.configs import (
    AdapterConfig,
    BackfeedingAdapterConfig,
    BackfeedingAdapterOnFeaturesConfig,
    CformerAdapterConfig,
    CifAdapterConfig,
    CmlpAdapterConfig,
    ConvolutionalAdapterConfig,
    CtcAdapterConfig,
    MlpAdapterConfig,
    TransformerAdapterConfig,
    WindowLevelQformerAdapterConfig,
)
from speechlmm.model.adapters.outputs import (
    BackfeedingAdapterOutput,
    CifAdapterOutput,
    CodecOutput,
    CtcAdapterOutput,
    SpeechLmmModuleOutput,
    WindowQformerOutput,
)
from speechlmm.model.adapters.qformer import BertConfig as QformerConfig
from speechlmm.model.adapters.qformer import BertModel as QformerModel
from speechlmm.model.adapters.utils import (
    Interpolation,
    PaddingAwareConv1d,
    TotalTrackingDict,
    count_trainable_parameters,
    lengths_to_padding_mask,
)
from speechlmm.model.attn_implementation import AttentionImplementationMixin
from speechlmm.model.embeddings import ScaledEmbedding


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


# Simple net blocks
# ------------------------------------------
class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class MLPBlock(nn.Module):  # NOTE added by pier
    """
    A simple MLP block with two linear layers and a ReLU activation function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        residual: bool = False,
    ) -> None:
        super(MLPBlock, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        if residual:
            raise NotImplementedError(
                "Residual connection is not implemented yet"
            )
        self.residual = residual

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Audio in output codec projection
# ------------------------------------------
class CodecFeatureProjectionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        per_channel_projection: bool = False,
        projector_type: str = "mlp",  # mlp, linear
        num_channels: int = 8,
        mlp_hidden_dim: int = 1024,
    ):
        super(CodecFeatureProjectionLayer, self).__init__()
        if projector_type == "mlp":
            self.projection_layer = MLPBlock(
                input_dim, mlp_hidden_dim, output_dim
            )
        elif projector_type == "linear":
            self.projection_layer = nn.Linear(input_dim, output_dim)
        else:
            raise NotImplementedError(
                f"Projector type {projector_type} is not implemented yet"
            )
        if per_channel_projection:
            assert (
                num_channels is not None
            ), "num_channels should be provided with per_channel_projection"
            self.projector = nn.ModuleList(
                [self.projection_layer for _ in range(output_dim)]
            )
        else:
            self.projector = self.projection_layer

        self.per_channel_projection = per_channel_projection

    def forward(self, x):
        if self.per_channel_projection:
            return torch.stack(
                [
                    projector(x["features_per_channel"][:, channel, :])
                    for channel, projector in enumerate(self.projector)
                ],
                dim=1,
            )
        else:
            return self.projector(x["features"])


# TODO(anferico): Rename to `SequenceAdapter` or something and also
# rename any `audio_encoder_output` param to `sequence_encoder_output`
# (rationale: adapters can adapt both audio and video sequences)
class AudioAdapter(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        config: AdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.granular_losses = ["lm_loss"]

    def get_trainable_parameters(
        self, return_plain_dict=True
    ) -> Union[Dict[str, Union[int, list, Dict]], TotalTrackingDict]:
        trainable_parameters = self._get_trainable_parameters()
        if not return_plain_dict:
            return trainable_parameters
        return trainable_parameters.to_dict()

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        # NOTE: this is the method that subclasses may override
        return TotalTrackingDict(total=count_trainable_parameters(self))

    @abstractmethod
    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> SpeechLmmModuleOutput:
        raise NotImplementedError(
            f"`forward()` has no implementation in {self.__class__.__name__}."
        )


class AttentionBasedAudioAdapter(AudioAdapter, AttentionImplementationMixin):
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(
        self,
        config: AdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)
        self.set_attn_implementation_with_fallback(attn_implementation)


class MlpAdapter(AudioAdapter):
    def __init__(
        self,
        config: MlpAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)

        # Optional residual connection
        if (
            self.config.residual_type != "none"
            and self.config.input_dim != self.config.output_dim
        ):
            if self.config.residual_type == "linear":
                self.residual_projector = nn.Linear(
                    self.config.input_dim,
                    self.config.output_dim,
                    dtype=self.torch_dtype,
                )
            elif self.config.residual_type == "interpolation":
                self.residual_projector = Interpolation(
                    size=self.config.output_dim,
                    mode="linear",
                    align_corners=False,
                )
            else:
                raise ValueError(
                    f"Unknown residual type '{self.config.residual_type}'."
                )

        self.needs_input_projector = (
            self.config.input_dim != self.config.hidden_size
            or self.config.force_input_projection
        )
        layers = []
        if self.needs_input_projector:
            layers.extend(
                [
                    nn.Linear(
                        self.config.input_dim,
                        self.config.hidden_size,
                        dtype=self.torch_dtype,
                    ),
                    nn.LayerNorm(
                        self.config.hidden_size, dtype=self.torch_dtype
                    ),
                    nn.ReLU(),
                    # ↑ NOTE: conceptually speaking, an activation
                    # function should not be used in a projection
                    # module. However, since this architecture is an
                    # MLP, we include it to avoid stacking 2 linear
                    # layers, which would be equivalent to a single one
                ]
            )

        for _ in range(self.config.hidden_layers):
            layers.extend(
                [
                    nn.Linear(
                        self.config.hidden_size,
                        self.config.hidden_size,
                        dtype=self.torch_dtype,
                    ),
                    nn.LayerNorm(
                        self.config.hidden_size, dtype=self.torch_dtype
                    ),
                    nn.ReLU(),
                ]
            )

        self.needs_output_projector = (
            self.config.hidden_size != self.config.output_dim
            or self.config.force_output_projection
        )
        if self.needs_output_projector:
            layers.extend(
                [
                    nn.Linear(
                        self.config.hidden_size,
                        self.config.output_dim,
                        dtype=self.torch_dtype,
                    ),
                    nn.LayerNorm(
                        self.config.output_dim, dtype=self.torch_dtype
                    ),
                ]
            )

        self.layers = nn.Sequential(*layers)

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        if self.needs_input_projector:
            trainable_params["input_projector"] = count_trainable_parameters(
                self.layers[0]
            ) + count_trainable_parameters(self.layers[1])
            first_hidden_linear_idx = 3
            # ↑ Input projector is composed of 3 layers, and the
            # first hidden Linear layer comes immediately after
        else:
            first_hidden_linear_idx = 0

        # LayerNorm layers always come after Linear layers
        first_hidden_lnorm_idx = first_hidden_linear_idx + 1

        if self.needs_output_projector:
            trainable_params["output_projector"] = count_trainable_parameters(
                self.layers[-2]
            ) + count_trainable_parameters(self.layers[-1])
            last_hidden_linear_idx = -5
            # ↑ Output projector is composed of 2 layers, and the
            # last hidden "block" is composed of 3 layers, of which
            # the first is a Linear layer
        else:
            last_hidden_linear_idx = -3

        # LayerNorm layers always come after Linear layers
        last_hidden_lnorm_idx = last_hidden_linear_idx + 1

        trainable_params["hidden_layers"] = [
            count_trainable_parameters(linear)
            + count_trainable_parameters(lnorm)
            for linear, lnorm in zip(
                self.layers[
                    first_hidden_linear_idx : last_hidden_linear_idx + 1 : 3
                ],
                self.layers[
                    first_hidden_lnorm_idx : last_hidden_lnorm_idx + 1 : 3
                ],
            )
        ]

        if hasattr(self, "residual_projector"):
            trainable_params["residual_projector"] = (
                count_trainable_parameters(self.residual_projector)
            )

        return trainable_params

    def forward(
        self,
        encoder_output: SpeechLmmModuleOutput,
        **kwargs,  # NOTE: added for retrocompatibility
    ) -> SpeechLmmModuleOutput:

        features = encoder_output.features

        projected_features = self.layers(features)
        if self.config.residual_type != "none":
            if hasattr(self, "residual_projector"):
                residual = self.residual_projector(features)
            else:
                residual = features
            projected_features = projected_features + residual

        return SpeechLmmModuleOutput(
            features=projected_features,
            padding_mask=encoder_output.padding_mask,
        )


class TransformerAdapter(AttentionBasedAudioAdapter):
    _supports_flash_attn_2 = BertModel._supports_flash_attn_2
    _supports_sdpa = BertModel._supports_sdpa

    def __init__(
        self,
        config: TransformerAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        bert_config = BertConfig(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            layer_norm_eps=self.config.layer_norm_eps,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
        )
        self.bert_encoder = BertModel(
            bert_config, add_pooling_layer=False
        ).encoder

        if self.config.input_dim != self.config.hidden_size:
            self.input_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_dim,
                    self.config.hidden_size,
                    dtype=self.torch_dtype,
                ),
                nn.LayerNorm(self.config.hidden_size, dtype=self.torch_dtype),
            )
        if self.config.output_dim != self.config.hidden_size:
            self.output_projector = nn.Sequential(
                nn.Linear(
                    self.config.hidden_size,
                    self.config.output_dim,
                    dtype=self.torch_dtype,
                ),
                nn.LayerNorm(self.config.output_dim, dtype=self.torch_dtype),
            )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        if hasattr(self, "input_projector"):
            trainable_params["input_projector"] = count_trainable_parameters(
                self.input_projector
            )
        trainable_params["bert_encoder"] = count_trainable_parameters(
            self.bert_encoder
        )
        if hasattr(self, "output_projector"):
            trainable_params["output_projector"] = count_trainable_parameters(
                self.output_projector
            )

        return trainable_params

    def forward(
        self,
        audio_encoder_output: SpeechLmmModuleOutput,
        **kwargs,  # NOTE: added for retrocompatibility
    ) -> SpeechLmmModuleOutput:

        audio_features = audio_encoder_output.features
        if hasattr(self, "input_projector"):
            audio_features = self.input_projector(audio_features)

        attention_mask = ~audio_encoder_output.padding_mask
        bert_encoder_output = self.bert_encoder(
            hidden_states=audio_features,
            attention_mask=attention_mask[:, None, None, :],
        )
        if hasattr(self, "output_projector"):
            audio_features = self.output_projector(
                bert_encoder_output.last_hidden_state
            )

        return SpeechLmmModuleOutput(
            features=audio_features,
            padding_mask=audio_encoder_output.padding_mask,
        )
        # ↑ NOTE: we reuse the attention mask from the audio encoder for
        # two reasons:
        # 1) the Transformer layers don't alter the input length
        # 2) some attention implementations (e.g. SDPA) don't support
        #    outputting attention weights, so we can't rely on them to
        #    generate a new attention mask


class WindowLevelQformerAdapter(AttentionBasedAudioAdapter):
    _supports_flash_attn_2 = QformerModel._supports_flash_attn_2
    _supports_sdpa = QformerModel._supports_sdpa

    def __init__(
        self,
        config: WindowLevelQformerAdapterConfig,
        audio_features_sampling_rate: float,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        if getattr(self.config, "triplet_loss", False):
            self.granular_losses.append("triplet_loss")

        self.pre_qformer_layer_norm = nn.LayerNorm(
            self.config.input_dim, dtype=self.torch_dtype
        )

        qformer_config = QformerConfig(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            layer_norm_eps=self.config.layer_norm_eps,
            add_cross_attention=self.config.add_cross_attention,
            cross_attention_hidden_size=self.config.input_dim,
            num_queries=self.config.num_queries,
            cross_attention_every_n_layers=self.config.cross_attention_every_n_layers,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
        )

        self.qformer = QformerModel(qformer_config).to(dtype=self.torch_dtype)

        self.query_embeds = nn.Parameter(
            torch.empty(
                1,
                self.config.num_queries,
                self.config.hidden_size,
                dtype=self.torch_dtype,
            ).normal_(mean=0.0, std=self.config.initializer_range)
        )
        # `self.query_embeds` is used in `self.qformer.forward()`
        # despite being owned by `self`, so we must explicitly instruct
        # DeepSpeed to coordinate its collection and partitioning in the
        # forward and backward passes of `self.qformer`
        deepspeed.zero.register_external_parameter(
            self.qformer, self.query_embeds
        )

        if self.config.output_dim != self.config.hidden_size:
            self.output_projector = nn.Linear(
                self.config.hidden_size,
                self.config.output_dim,
                dtype=self.torch_dtype,
            )

        self.audio_features_sampling_rate = audio_features_sampling_rate

        del self.qformer.embeddings.word_embeddings
        del self.qformer.embeddings.position_embeddings
        for layer in self.qformer.encoder.layer:
            # NOTE: these are deleted as they are replaced by two other
            # layers called `query_intermediate` and `query_output`
            # (cf. speechlmm/model/adapters/qformer/BertLayer)
            del layer.intermediate
            del layer.output

        if self.config.compress_factor is None:
            self.config.compress_factor = round(
                self.audio_features_sampling_rate
                * self.config.window_size_in_seconds
            )
            print(
                f"""Compression factor is not specified, found window_size_in_seconds: {self.config.window_size_in_seconds}.
                    The window_size_in_seconds is deprecated and will be removed in the future.
                    Setting compress_factor to {self.config.compress_factor}.
                    This will be used to set the window size,
                    replacing the `window_size_in_seconds` parameter."""
            )

        else:
            print(
                f"""Compression factor is set to {self.config.compress_factor}.
                  The resulting window size in seconds is: {self.config.compress_factor / self.audio_features_sampling_rate}.
                  If you want to use the `window_size_in_seconds` parameter,
                  please remove the `compress_factor` parameter from the configuration,
                  but be aware that the `window_size_in_seconds` parameter is deprecated and will be removed in the future."""
            )

    def forward(
        self,
        audio_encoder_output: SpeechLmmModuleOutput,
        **kwargs,  # NOTE: added for retrocompatibility
    ) -> SpeechLmmModuleOutput:
        input_audio_features = audio_encoder_output.features
        cross_attention_mask = ~audio_encoder_output.padding_mask
        # NOTE: `cross_attention_mask` is used in the cross-attention to
        # avoid attending to audio features corresponding to padding
        output_audio_features = self.pre_qformer_layer_norm(
            input_audio_features
        )
        # check if duration present in kwargs
        durations = kwargs.get("durations", None)
        inputs_to_split = {
            "embeds": output_audio_features,
            "attention_mask": cross_attention_mask,
        }
        split_fn = self.split_into_windows
        if durations is not None:
            split_fn = self.split_into_windows_with_durations
            inputs_to_split["durations"] = durations

        split_fn_output = split_fn(**inputs_to_split)
        windowed_audio_features = split_fn_output["embeds"]
        windowed_cross_attention_mask = split_fn_output["attention_mask"]
        hop_length = split_fn_output.get("hop_length", 0)

        batch_size_times_num_windows, _, _ = windowed_audio_features.shape
        query_embeds = self.query_embeds.expand(
            batch_size_times_num_windows, -1, -1
        )

        # NOTE: we don't pass `attention_mask` because every query must
        # attend to each other. Padded positions only ever exist in the
        # input sequence, and are handled by `cross_attention_mask`
        query_output = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=windowed_audio_features,
            encoder_attention_mask=windowed_cross_attention_mask,
            return_dict=True,
        )
        output_audio_features = query_output.last_hidden_state

        if hasattr(self, "output_projector"):
            output_audio_features = self.output_projector(
                output_audio_features
            )

        batch_size, _, _ = input_audio_features.shape
        _, _, emb_dim = output_audio_features.shape
        output_audio_features = output_audio_features.view(
            batch_size, -1, emb_dim
        ).contiguous()
        query_attention_mask = (
            (
                windowed_cross_attention_mask[:, hop_length:].sum(
                    dim=1, keepdim=True
                )
                > 0
            )
            .repeat(1, self.config.num_queries)
            .to(dtype=torch.bool, device=query_embeds.device)
        )
        query_attention_mask = query_attention_mask.view(
            batch_size, -1
        ).contiguous()

        return WindowQformerOutput(
            features=output_audio_features,
            padding_mask=~query_attention_mask,
            windowed_cross_attention_mask=windowed_cross_attention_mask[
                :, hop_length:
            ].contiguous(),
        )

    def split_into_windows(self, embeds, attention_mask, truncate_last=False):
        batch_size, seq_len, emb_dim = embeds.shape

        # NOTE (pier) It seems to me that the `compress_factor` is
        # the best way to control the window size, as we reason in terms of compression
        # For this reason, I suggest to use the compress_factor as the main parameter
        # to control the window size, and to deprecate the `window_size_in_seconds` parameter
        # For now, for retrocompatibility the `window_size_in_seconds` is still used, but
        # it is replaced by the `compress_factor` parameter
        window_size_in_frames = self.config.compress_factor

        # TODO(anferico): here and further below (cf. `unfold`) we
        # select `stride` (a.k.a. hop length) to be the same as
        # `kernel_size` (a.k.a. window size), as the expression for
        # `excess_frames_in_last_window` reveals. Ideally, we should
        # allow for independent values (e.g. by adding a new
        # configuration parameter)
        excess_frames_in_last_window = seq_len % window_size_in_frames
        if not truncate_last and excess_frames_in_last_window > 0:
            # Zero-pad along the seq_len dimension
            pad_length = window_size_in_frames - excess_frames_in_last_window
            embeds = F.pad(embeds, (0, 0, 0, pad_length))
            attention_mask = F.pad(attention_mask, (0, pad_length))

        # NOTE: the input to `unfold` must have the spatial dimensions
        # as the last dimensions, and they must be at least 2. So, we
        # reshape `embeds` and add a dummy dimension as well.
        new_size_in_frames = int(
            round(window_size_in_frames * (1 + self.config.hop_size))
        )
        hop_length = new_size_in_frames - window_size_in_frames
        stride = (1, window_size_in_frames)
        padding = (0, hop_length)
        kernel_size = (
            1,
            new_size_in_frames,
        )
        unfold_input = embeds.transpose(1, 2).unsqueeze(2)
        windowed_embeds = F.unfold(
            unfold_input,  # (batch_size, emb_dim, 1, seq_len)
            stride=stride,
            padding=padding,
            kernel_size=kernel_size,
        )
        unfold_attn = attention_mask.unsqueeze(1).float()
        windowed_attn_mask = torch.nn.functional.unfold(
            unfold_attn,  # (batch_size, emb_dim, 1, seq_len)
            kernel_size=kernel_size,  # (1, 6),
            stride=stride,  # (1, 4),
            padding=padding,  # (0, 2),
        )
        _, _, num_windows = windowed_embeds.shape
        windowed_embeds = windowed_embeds.view(
            batch_size, emb_dim, kernel_size[1], num_windows
        )
        windowed_attn = windowed_attn_mask.view(
            batch_size, new_size_in_frames, num_windows
        )
        windowed_embeds = torch.permute(windowed_embeds, [0, 3, 2, 1])
        embeds = windowed_embeds.reshape(-1, kernel_size[1], emb_dim)
        # ↑ (batch_size * num_windows, window_size_in_frames + hop size, emb_dim)
        windowed_attn = torch.permute(windowed_attn, [0, 2, 1])
        attention_mask = windowed_attn.reshape(-1, new_size_in_frames)

        if truncate_last:
            if self.config.hop_size:
                raise NotImplementedError(
                    "truncate_last not implemented with hop_size"
                )
            attention_mask = attention_mask[
                :, : seq_len - seq_len % window_size_in_frames
            ].contiguous()

        return {
            "embeds": embeds,
            "attention_mask": attention_mask.bool(),
            "hop_length": hop_length,
        }

    def _pad_features_list_and_attention(
        self, features_list, attention_list, exclude_empty_attentions=False
    ):
        """
        Padding the features in case of duration provided
        """
        # FIXME check if there is another value to pad with
        padded_list = []
        attention_masks = []
        pad_tensor = (
            torch.zeros(1, self.feature_d)
            .to(features_list[0].device)
            .to(features_list[0].dtype)
        )
        max_len = max([t.size(0) for t in features_list])
        for t, a in zip(features_list, attention_list):
            if exclude_empty_attentions and not a.sum():
                raise NotImplementedError(
                    "exclude_empty_attentions not implemented yet"
                )
            if t.size(0) < max_len:
                padded_list += [
                    torch.cat(
                        [t, pad_tensor.repeat(max_len - t.size(0), 1)], dim=0
                    )
                ]
                attention_masks += [
                    torch.cat(
                        [a, torch.zeros(max_len - t.size(0)).to(a.device)],
                        dim=0,
                    )
                ]
            else:
                padded_list += [t]
                attention_masks += [a]
        return (
            torch.stack(padded_list, dim=0),
            torch.stack(attention_masks, dim=0).bool(),
        )

    def split_into_windows_with_durations(
        self, embeds, attention_mask, durations, exclude_empty_attentions=False
    ):
        features_list = []
        attentions = []
        for sample, sample_mask, sample_durations in zip(
            embeds, attention_mask, durations
        ):
            start = 0
            for d in sample_durations:
                features_list.append(sample[start : start + d, :])
                attentions.append(sample_mask[start : start + d])
                start += d
        return self._pad_features_list_and_attention(
            features_list, attentions, exclude_empty_attentions
        )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()

        trainable_parameters_in_queries = (
            self.query_embeds.numel() if self.query_embeds.requires_grad else 0
        )
        trainable_params["qformer"] = (
            trainable_parameters_in_queries
            + count_trainable_parameters(self.pre_qformer_layer_norm)
            + count_trainable_parameters(self.qformer)
        )

        if hasattr(self, "output_projector"):
            trainable_params["output_projector"] = count_trainable_parameters(
                self.output_projector
            )

        return trainable_params


class ConvolutionalAdapter(AudioAdapter):
    def __init__(
        self,
        config: ConvolutionalAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)

        self.conv1d_in = PaddingAwareConv1d(
            in_channels=self.config.input_dim,
            out_channels=self.config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=self.torch_dtype,
        )
        self.layer_norm_in = nn.LayerNorm(
            self.config.hidden_size, dtype=self.torch_dtype
        )
        self.residual_conv1d_hidden_layers = nn.ModuleList(
            [
                PaddingAwareConv1d(
                    in_channels=self.config.hidden_size,
                    out_channels=self.config.hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.config.hidden_layers)
            ]
        )
        self.layer_norm_hidden_layers = nn.ModuleList(
            [
                nn.LayerNorm(self.config.hidden_size, dtype=self.torch_dtype)
                for _ in range(self.config.hidden_layers)
            ]
        )
        self.conv1d_out = PaddingAwareConv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.output_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=self.torch_dtype,
        )
        self.layer_norm_out = nn.LayerNorm(
            self.config.output_dim, dtype=self.torch_dtype
        )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        trainable_params["input_projector"] = count_trainable_parameters(
            self.conv1d_in
        ) + count_trainable_parameters(self.layer_norm_in)

        trainable_params["hidden_layers"] = [
            count_trainable_parameters(conv)
            + count_trainable_parameters(lnorm)
            for conv, lnorm in zip(
                self.residual_conv1d_hidden_layers,
                self.layer_norm_hidden_layers,
            )
        ]

        trainable_params["output_projector"] = count_trainable_parameters(
            self.conv1d_out
        ) + count_trainable_parameters(self.layer_norm_out)

        return trainable_params

    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> SpeechLmmModuleOutput:
        features = audio_encoder_output.features
        padding_mask = audio_encoder_output.padding_mask

        # First convolution (stride=2)
        projected_features, padding_mask = self.conv1d_in(
            features.permute((0, 2, 1)), padding_mask
        )
        projected_features = self.layer_norm_in(
            projected_features.permute((0, 2, 1))
        )
        projected_features = F.relu(projected_features)

        # Intermediate convolutions (stride=1, w/ residual connections)
        for conv, layer_norm in zip(
            self.residual_conv1d_hidden_layers, self.layer_norm_hidden_layers
        ):
            residual = projected_features
            projected_features, padding_mask = conv(
                projected_features.permute((0, 2, 1)), padding_mask
            )
            projected_features = layer_norm(
                projected_features.permute((0, 2, 1))
            )
            projected_features = F.relu(projected_features) + residual

        # Last convolution (stride=2)
        projected_features, padding_mask = self.conv1d_out(
            projected_features.permute((0, 2, 1)), padding_mask
        )
        projected_features = self.layer_norm_out(
            projected_features.permute((0, 2, 1))
        )

        return SpeechLmmModuleOutput(
            features=projected_features, padding_mask=padding_mask
        )


class CifAdapter(AudioAdapter):
    def __init__(
        self,
        config: CifAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)

        if self.config.quantity_loss_weight > 0:
            self.granular_losses.append("quantity_loss")
        if self.config.ctc_loss_weight > 0:
            self.granular_losses.append("ctc_loss")
            self.ctc_projection = nn.Linear(
                self.config.input_dim,
                self.config.ctc_loss_vocab_size,
                dtype=self.torch_dtype,
            )

        self.input_weights_extractor = nn.Sequential(
            torchvision.ops.Permute((0, 2, 1)),
            # ↑ (batch_size, T, C) -> (batch_size, C, T) as required by 1D convolution
            torch.nn.Conv1d(
                in_channels=self.config.input_dim,
                out_channels=self.config.input_dim,
                kernel_size=self.config.conv_kernel_size,
                stride=self.config.conv_stride,
                padding=self.config.conv_padding,
                dtype=self.torch_dtype,
            ),
            torchvision.ops.Permute((0, 2, 1)),
            # ↑ (batch_size, C, T) -> (batch_size, T, C) after 1D convolution
            torch.nn.LayerNorm(self.config.input_dim, dtype=self.torch_dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=self.config.input_dim,
                out_features=1,  # scalar weights for each input feature
                dtype=self.torch_dtype,
            ),
            torch.nn.Sigmoid(),
        )

        self.cif = Cif(**self.config.cif)

        if self.config.output_dim != self.config.input_dim:
            self.output_projector = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.output_dim),
                nn.LayerNorm(self.config.output_dim),
            )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        if hasattr(self, "ctc_projection"):
            trainable_params["ctc_projection"] = count_trainable_parameters(
                self.ctc_projection
            )

        trainable_params["input_weights_extractor"] = (
            count_trainable_parameters(self.input_weights_extractor)
        )

        if hasattr(self, "output_projector"):
            trainable_params["output_projector"] = count_trainable_parameters(
                self.output_projector
            )

        return trainable_params

    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> CifAdapterOutput:
        input_embeddings = audio_encoder_output.features
        input_weights = self.input_weights_extractor(input_embeddings).squeeze(
            -1
        )
        padding_mask = audio_encoder_output.padding_mask
        targets_attention_mask = kwargs.get("transcription_attention_mask")
        SEQ_LEN_DIM = 1
        target_lengths = (
            targets_attention_mask.sum(SEQ_LEN_DIM)
            if targets_attention_mask is not None
            else None
        )
        cif_output = self.cif(
            inputs=input_embeddings,
            input_weights=input_weights,
            padding_mask=padding_mask,
            target_lengths=target_lengths,
        )

        projector_output = dict()
        if hasattr(self, "output_projector"):
            projector_output["features"] = self.output_projector(
                cif_output["integrated_embeddings"]
            )
        else:
            projector_output["features"] = cif_output["integrated_embeddings"]

        projector_output["padding_mask"] = lengths_to_padding_mask(
            cif_output["integrated_embeddings_lengths"]
        )

        if target_lengths is not None:
            if self.config.quantity_loss_weight > 0:
                projector_output["quantity_loss"] = torch.abs(
                    cif_output["input_weights_sum"] - target_lengths
                ).mean()
            if self.config.ctc_loss_weight > 0:
                ctc_logits = self.ctc_projection(input_embeddings)
                ctc_logprobs = F.log_softmax(ctc_logits, dim=-1)
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        input_embeddings.size(0),
                        input_embeddings.size(1),
                        dtype=torch.bool,
                    )
                targets = kwargs["transcription_ids"]
                input_lengths = (~padding_mask).sum(SEQ_LEN_DIM)
                with torch.backends.cudnn.flags(enabled=False):
                    # ↑ Disable CuDNN while computing the CTC loss as
                    # the CuDNN implementation imposes additional
                    # constraints on the input arguments.
                    # cf. https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
                    projector_output["ctc_loss"] = F.ctc_loss(
                        log_probs=(
                            ctc_logprobs.permute(1, 0, 2).float().contiguous()
                        ),
                        # ↑ ctc_loss expects:
                        # 1) shape = (seq_len, batch_size, vocab_size)
                        # 2) dtype = float32
                        targets=targets,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.config.ctc_loss_blank_id,
                        reduction="mean",
                        zero_infinity=True,
                    ).to(dtype=input_embeddings.dtype)

        return CifAdapterOutput(**projector_output)


class CtcAdapter(AudioAdapter):
    def __init__(
        self,
        config: CtcAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)

        self.granular_losses.append("ctc_loss")

        self.ctc_logprobs_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "logits_extractor",
                        nn.Linear(
                            self.config.input_dim,
                            self.config.ctc_loss_vocab_size,
                            dtype=self.torch_dtype,
                        ),
                    ),
                    ("log_softmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )
        if self.config.output_dim != self.config.input_dim:
            self.output_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_dim,
                    self.config.output_dim,
                    dtype=self.torch_dtype,
                ),
                nn.LayerNorm(self.config.output_dim, dtype=self.torch_dtype),
            )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        trainable_params["ctc_logprobs_extractor"] = (
            count_trainable_parameters(self.ctc_logprobs_extractor)
        )

        if hasattr(self, "output_projector"):
            trainable_params["output_projector"] = count_trainable_parameters(
                self.output_projector
            )

        return trainable_params

    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> CtcAdapterOutput:
        input_embeddings = audio_encoder_output.features

        ctc_logprobs = self.ctc_logprobs_extractor(input_embeddings)

        padding_mask = audio_encoder_output.padding_mask
        SEQ_LEN_DIM = 1
        input_lengths = (~padding_mask).sum(SEQ_LEN_DIM)
        targets_attention_mask = kwargs.get("transcription_attention_mask")
        ctc_loss = None
        if targets_attention_mask is not None:  # training time
            with torch.backends.cudnn.flags(enabled=False):
                # ↑ Disable CuDNN while computing the CTC loss, as the
                # CuDNN implementation imposes additional constraints on
                # the input arguments.
                # cf. https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
                targets = kwargs["transcription_ids"]
                blank_id = self.config.ctc_loss_blank_id
                # ↑ TODO: rename cif_ctc_loss_blank_id
                targets[~targets_attention_mask] = blank_id
                target_lengths = targets_attention_mask.sum(SEQ_LEN_DIM)
                ctc_loss = F.ctc_loss(
                    log_probs=(
                        ctc_logprobs.permute(1, 0, 2).float().contiguous()
                    ),
                    # ↑ ctc_loss expects:
                    # 1) shape = (seq_len, batch_size, vocab_size)
                    # 2) dtype = float32
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=True,
                ).to(dtype=input_embeddings.dtype)

        with torch.no_grad():
            compressed_input_embeddings, padding_mask_post_compression = (
                self._ctc_compress_sequence(
                    input_embeddings,
                    padding_mask,
                    ctc_logprobs,
                    input_lengths=input_lengths,
                )
            )

        if hasattr(self, "output_projector"):
            compressed_input_embeddings = self.output_projector(
                compressed_input_embeddings
            )

        return CtcAdapterOutput(
            features=compressed_input_embeddings,
            padding_mask=padding_mask_post_compression,
            ctc_loss=ctc_loss,
        )

    def _ctc_compress_sequence(
        self,
        sequence,
        padding_mask,
        ctc_logprobs,
        input_lengths: torch.LongTensor = None,
    ):
        # filter out positions corresponding to padding tokens, then get
        # the predicted symbols
        predicted_symbols = [
            logprobs[~mask].argmax(dim=-1)
            for logprobs, mask in zip(ctc_logprobs, padding_mask)
        ]
        # group the predicted symbols into runs of the same symbol
        # e.g.: ["a", "b", "b", "a"] -> [("a", 1), ("b", 2), ("a", 1)]
        grouped_predicted_symbols = [
            [
                (symbol, len(list(symbol_run)))
                for symbol, symbol_run in itertools.groupby(symbols)
            ]
            for symbols in predicted_symbols
        ]

        if (
            self.config.max_compressed_length is not None
            and input_lengths is not None
        ):
            # https://aclanthology.org/2022.iwslt-1.13.pdf (Section 2.1)
            grouped_predicted_symbols = (
                self._precompress_sequence_to_avoid_oom(
                    grouped_predicted_symbols, input_lengths
                )
            )

        seq_lens_post_compression = [len(p) for p in grouped_predicted_symbols]
        # ↑ NOTE: this works under the assumption that <blank> tokens
        # are NOT removed (as in the case of the original paper)
        max_seq_len_post_compression = max(seq_lens_post_compression)
        batch_size, max_seq_len = ctc_logprobs.shape[:2]
        compression_matrix = torch.zeros(
            (batch_size, max_seq_len, max_seq_len_post_compression),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        for batch_index, groups in enumerate(grouped_predicted_symbols):
            n_consumed_symbols = 0
            for seq_index, group in enumerate(groups):
                _, symbol_run_len = group
                compression_matrix[
                    batch_index,
                    n_consumed_symbols : (n_consumed_symbols + symbol_run_len),
                    seq_index,
                ] = self._get_weight_for_group(
                    group, ctc_logprobs, batch_index, n_consumed_symbols
                )
                n_consumed_symbols += symbol_run_len

        # compression_matrix: (batch_size, T, T')
        # input_embeddings: (batch_size, T, C) -> (batch_size, C, T)
        # compressed_input_embeddings: (batch_size, C, T') -> (batch_size, T', C)
        compressed_sequence = (
            sequence.permute(0, 2, 1).bmm(compression_matrix).permute(0, 2, 1)
        )
        padding_mask_post_compression = lengths_to_padding_mask(
            torch.tensor(seq_lens_post_compression, dtype=torch.long)
        ).to(device=padding_mask.device)
        return compressed_sequence, padding_mask_post_compression

    def _get_weight_for_group(
        self, group, ctc_logprobs, batch_index, n_consumed_symbols
    ):
        symbol, symbol_run_len = group
        compression_strategy = self.config.compression_strategy
        if compression_strategy == "mean":
            return 1.0 / symbol_run_len
        elif compression_strategy in ["weighted", "softmax"]:
            weights = ctc_logprobs[
                batch_index,
                n_consumed_symbols : (n_consumed_symbols + symbol_run_len),
                symbol,
            ]
            if compression_strategy == "softmax":
                weights = F.softmax(weights)
            return weights / weights.sum()
        else:
            raise ValueError(
                f"Unknown CTC compression strategy: {compression_strategy}"
            )

    def _precompress_sequence_to_avoid_oom(
        self,
        batch_predicted,
        input_lengths: torch.LongTensor,
    ):
        """
        Ensures that the output of the CTC compression is not longer than the ctc_compress_max_out_size.
        If there are samples violating this constraints, consecutive predictions are merged
        so to shorten the sentence.
        E.g. if the ctc_compress_max_out_size is set to 3, and the output of the CTC compression would be
        long 5, the first and second predictions are merged, as well as the third and the fourth. So, the
        corresponding vectors will be merged according to the CTC compression strategy.
        """

        def merge_sublist(elements):
            """
            Takes a list of Tuples (predicted_element, num_corresponding_vectors) and returns
            a single tuple with the predicted_element having the highest number of corresponding_vectors
            (in case of a tie, the first is returned) and the total sum of the num_corresponding_vectors
            E.g. if the input is [(a, 3), (b, 5), (c, 6), (a, 4)], the output will be (a, 18).
            """
            sum_num_vectors = 0
            max_element = None
            max_element_cnt = 0
            temp_dict = {}
            for predicted_element, num_corresponding_vectors in elements:
                if predicted_element in temp_dict:
                    temp_dict[predicted_element] += num_corresponding_vectors
                else:
                    temp_dict[predicted_element] = num_corresponding_vectors
                if temp_dict[predicted_element] > max_element_cnt:
                    max_element_cnt = temp_dict[predicted_element]
                    max_element = predicted_element
                sum_num_vectors += num_corresponding_vectors
            return max_element, sum_num_vectors

        max_compressed_length = self.config.max_compressed_length
        if isinstance(max_compressed_length, float):
            ctc_compress_max_out_sizes = (
                (input_lengths * max_compressed_length).long().tolist()
            )
        else:
            ctc_compress_max_out_sizes = [max_compressed_length] * len(
                input_lengths
            )
        for b_idx, (p, max_len) in enumerate(
            zip(batch_predicted, ctc_compress_max_out_sizes)
        ):
            pred_len = len(p)
            if pred_len > max_len:
                reduction_factor = math.ceil(pred_len / max_len)
                i = 0
                new_p = []
                while i < pred_len:
                    new_p.append(merge_sublist(p[i : i + reduction_factor]))
                    i += reduction_factor
                batch_predicted[b_idx] = new_p

        return batch_predicted


# TODO(anferico): extract common behavior between Cformer and Cmlp into
# a base class (they are almost identical)
class CformerAdapter(AttentionBasedAudioAdapter):
    _supports_flash_attn_2 = TransformerAdapter._supports_flash_attn_2
    _supports_sdpa = TransformerAdapter._supports_sdpa

    def __init__(
        self,
        config: CformerAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # Length adapter
        if self.config.length_adapter.model_type == "conv":
            self.length_adapter = ConvolutionalAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        elif self.config.length_adapter.model_type == "cif":
            self.length_adapter = CifAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        elif self.config.length_adapter.model_type == "ctc":
            self.length_adapter = CtcAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        else:
            raise ValueError(
                f"Unknown length adapter "
                f"'{self.config.length_adapter.model_type}'."
            )

        self.granular_losses = self.length_adapter.granular_losses

        # Modality adapter
        if hasattr(self.config, "modality_adapter_before"):
            self.modality_adapter_before = TransformerAdapter(
                self.config.modality_adapter_before,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
            )
        if hasattr(self.config, "modality_adapter_after"):
            self.modality_adapter_after = TransformerAdapter(
                self.config.modality_adapter_after,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
            )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        if hasattr(self, "modality_adapter_before"):
            trainable_params["modality_adapter_before"] = (
                count_trainable_parameters(self.modality_adapter_before)
            )

        trainable_params["length_adapter"] = (
            self.length_adapter.get_trainable_parameters(
                return_plain_dict=False
            )
        )

        if hasattr(self, "modality_adapter_after"):
            trainable_params["modality_adapter_after"] = (
                count_trainable_parameters(self.modality_adapter_after)
            )

        return trainable_params

    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> SpeechLmmModuleOutput:
        length_adapter_input = audio_encoder_output
        if hasattr(self, "modality_adapter_before"):
            length_adapter_input = self.modality_adapter_before(
                audio_encoder_output
            )

        length_adapter_output = self.length_adapter(
            length_adapter_input, **kwargs
        )

        if hasattr(self, "modality_adapter_after"):
            modality_adapter_after_output = self.modality_adapter_after(
                length_adapter_output
            )
            length_adapter_output.features = (
                modality_adapter_after_output.features
            )
            # NOTE: we return the very `length_adapter_output` with its
            #    `audio_features` replaced by that of the modality
            #    adapter because:
            # 1) the padding mask can be reused from the length adapter
            #    as Transformer layers don't alter the length of the
            #    input sequence
            # 2) we need to preserve the `granular_losses` field from
            #    the output of the length adapter
            # TODO(anferico): a more elegant solution would be to carry
            #    the `granular_losses` field through each module (audio
            #    encoder, modality adapter, length adapter, etc.)

        return length_adapter_output


class CMlpAdapter(AudioAdapter):
    def __init__(
        self,
        config: CmlpAdapterConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, torch_dtype=torch_dtype)

        # Length adapter
        if self.config.length_adapter.model_type == "conv":
            self.length_adapter = ConvolutionalAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        elif self.config.length_adapter.model_type == "cif":
            self.length_adapter = CifAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        elif self.config.length_adapter.model_type == "ctc":
            self.length_adapter = CtcAdapter(
                self.config.length_adapter,
                torch_dtype=self.torch_dtype,
            )
        else:
            raise ValueError(
                f"Unknown length adapter "
                f"'{self.config.length_adapter.model_type}'."
            )

        self.granular_losses = self.length_adapter.granular_losses

        # Modality adapter
        if hasattr(self.config, "modality_adapter_before"):
            self.modality_adapter_before = MlpAdapter(
                self.config.modality_adapter_before,
                torch_dtype=self.torch_dtype,
            )
        if hasattr(self.config, "modality_adapter_after"):
            self.modality_adapter_after = MlpAdapter(
                self.config.modality_adapter_after,
                torch_dtype=self.torch_dtype,
            )

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        trainable_params = TotalTrackingDict()
        if hasattr(self, "modality_adapter_before"):
            trainable_params["modality_adapter_before"] = (
                self.modality_adapter_before.get_trainable_parameters(
                    return_plain_dict=False
                )
            )

        trainable_params["length_adapter"] = (
            self.length_adapter.get_trainable_parameters(
                return_plain_dict=False
            )
        )

        if hasattr(self, "modality_adapter_after"):
            trainable_params["modality_adapter_after"] = (
                self.modality_adapter_after.get_trainable_parameters(
                    return_plain_dict=False
                )
            )

        return trainable_params

    def forward(
        self, audio_encoder_output: SpeechLmmModuleOutput, **kwargs
    ) -> SpeechLmmModuleOutput:
        length_adapter_input = audio_encoder_output
        if hasattr(self, "modality_adapter_before"):
            length_adapter_input = self.modality_adapter_before(
                audio_encoder_output
            )

        length_adapter_output = self.length_adapter(
            length_adapter_input, **kwargs
        )

        if hasattr(self, "modality_adapter_after"):
            modality_adapter_after_output = self.modality_adapter_after(
                length_adapter_output
            )
            length_adapter_output.features = (
                modality_adapter_after_output.features
            )
            # NOTE: we return the very `length_adapter_output` with its
            #    `audio_features` replaced by that of the modality
            #    adapter because:
            # 1) the padding mask can be reused from the length adapter
            #    as Transformer layers don't alter the length of the
            #    input sequence
            # 2) we need to preserve the `granular_losses` field from
            #    the output of the length adapter
            # TODO(anferico): a more elegant solution would be to carry
            #    the `granular_losses` field through each module (audio
            #    encoder, modality adapter, length adapter, etc.)

        return length_adapter_output


class BackfeedingAdapter(nn.Module):
    def __init__(
        self,
        config: BackfeedingAdapterConfig,
        audio_features_sampling_rate: float,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        audio_adapter_config = config.audio_adapter
        self.adapter = build_audio_adapter(
            audio_features_sampling_rate=audio_features_sampling_rate,
            attn_implementation=attn_implementation,
            config_dict=audio_adapter_config,
            torch_dtype=torch_dtype,
        )

    def get_trainable_parameters(
        self, return_plain_dict=True
    ) -> Union[Dict[str, Union[int, list, Dict]], TotalTrackingDict]:
        trainable_parameters = self._get_trainable_parameters()
        if not return_plain_dict:
            return trainable_parameters
        return trainable_parameters.to_dict()

    def _get_trainable_parameters(self) -> TotalTrackingDict:
        # NOTE: this is the method that subclasses may override
        return TotalTrackingDict(total=count_trainable_parameters(self))


class BackfeedingAdapterOnFeatures(BackfeedingAdapter):
    def __init__(
        self,
        config: BackfeedingAdapterOnFeaturesConfig,
        audio_features_sampling_rate: float,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            config=config,
            audio_features_sampling_rate=audio_features_sampling_rate,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.features_projector = None  # FIXME(pier-maker92): add projector
        self.norm_layer = nn.LayerNorm(4096)  # FIXME hardcoded

    def forward(
        self, codec_output: CodecOutput, **kwargs
    ) -> BackfeedingAdapterOutput:
        if self.features_projector is not None:
            audio_features_per_codebook = (
                codec_output.audio_features_per_codebook
            )
            audio_features = self.features_projector(
                audio_features_per_codebook
            )
            codec_output.features = audio_features
        adapter_output = self.adapter(codec_output)
        audio_emb = adapter_output.features  # 8, S, 4096
        # we get also the adapted padding mask, to handle fixed compression in which the padding mask is not the same as the original one
        adapted_padding_mask = adapter_output.padding_mask  # 8, S
        windowed_cross_attention_mask = getattr(
            adapter_output, "windowed_cross_attention_mask", None
        )
        text_embeddings = kwargs["text_embeddings"]
        if kwargs.get("training", True):
            # audio_padding_mask = codec_output.padding_mask  # 8, S
            new_audio_emb = []
            new_padding_mask = []
            for cur_audio_emb, cur_audio_padding_mask in zip(
                audio_emb, adapted_padding_mask
            ):
                # remove padding and last frame
                tail = 1 * getattr(self.adapter.config, "num_queries", 1)
                new_audio_emb.append(
                    cur_audio_emb[~cur_audio_padding_mask][:-tail, :]
                )
                # ⬇️ NOTE: (pier)
                # no reason to remove padding from the new adapted attention mask
                # as the padding express the total number of frames in the original audio
                # which will match an equal number of text tokens.
                new_padding_mask.append(
                    torch.zeros(
                        (~cur_audio_padding_mask).sum(-1) - tail + 1,
                        dtype=torch.bool,
                    )
                )
            audio_emb = torch.nn.utils.rnn.pad_sequence(
                new_audio_emb, batch_first=True
            )
            new_padding_mask = torch.nn.utils.rnn.pad_sequence(
                new_padding_mask, batch_first=True, padding_value=True
            )

            # add text and audio embeddings
            if kwargs.get("repeat_padding", False):
                text_embeddings = text_embeddings.repeat(
                    audio_emb.size(0), audio_emb.size(1), 1
                )
            else:
                text_embeddings = text_embeddings[:, 1:, :]
        else:
            new_padding_mask = adapted_padding_mask
        emb = (
            text_embeddings + audio_emb
        )  # (pier-maker92): adding a normalization layer?
        # emb = audio_emb

        codec_output.features = emb
        output = {
            "features": emb,
            "codes": codec_output.codes,
            "padding_mask": codec_output.padding_mask,
            "adapted_padding_mask": new_padding_mask,
        }
        if windowed_cross_attention_mask is not None:
            output["windowed_cross_attention_mask"] = (
                windowed_cross_attention_mask
            )

        return BackfeedingAdapterOutput(**output)


class BackfeedingAdapterOnCodes(BackfeedingAdapter):
    def __init__(
        self,
        config: BackfeedingAdapterOnFeaturesConfig,
        audio_features_sampling_rate: float,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        # FIXME(st3p99)
        super().__init__(
            config=config,
            audio_features_sampling_rate=audio_features_sampling_rate,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm="rms_norm",
            zero_idx=IGNORE_INDEX,
        )

        # lut_input_dim=self.codebook_size+1
        self.lut = nn.ModuleList(
            [EmbeddingFactory(2048, 4096) for _ in range(8)]
        )

    def forward(self, codec_output: CodecOutput, **kwargs) -> CodecOutput:
        audio_emb = [
            self.lut[i](codec_output.codes[:, :, i]) for i in range(8)
        ]
        audio_emb = torch.stack(audio_emb, dim=-2).sum(dim=-2)
        text_embeddings = kwargs["text_embeddings"]
        emb = text_embeddings + audio_emb
        codec_output.features = emb
        adapted_padding_mask = codec_output.padding_mask

        if kwargs.get("training", True):
            # audio_padding_mask = codec_output.padding_mask  # 8, S
            new_audio_emb = []
            new_padding_mask = []
            for cur_audio_emb, cur_audio_padding_mask in zip(
                audio_emb, adapted_padding_mask
            ):
                # remove padding and last frame
                tail = 1 * getattr(self.adapter.config, "num_queries", 1)
                new_audio_emb.append(
                    cur_audio_emb[~cur_audio_padding_mask][:-tail, :]
                )
                # ⬇️ NOTE: (pier)
                # no reason to remove padding from the new adapted attention mask
                # as the padding express the total number of frames in the original audio
                # which will match an equal number of text tokens.
                new_padding_mask.append(
                    torch.zeros(
                        (~cur_audio_padding_mask).sum(-1) - tail + 1,
                        dtype=torch.bool,
                    )
                )
            audio_emb = torch.nn.utils.rnn.pad_sequence(
                new_audio_emb, batch_first=True
            )
            new_padding_mask = torch.nn.utils.rnn.pad_sequence(
                new_padding_mask, batch_first=True, padding_value=True
            )

            # add text and audio embeddings
            if kwargs.get("repeat_padding", False):
                text_embeddings = text_embeddings.repeat(
                    audio_emb.size(0), audio_emb.size(1), 1
                )
            else:
                text_embeddings = text_embeddings[:, 1:, :]
        else:
            new_padding_mask = adapted_padding_mask
        emb = (
            text_embeddings + audio_emb
        )  # (pier-maker92): adding a normalization layer?
        # emb = audio_emb
        output = {
            "features": emb,
            "codes": codec_output.codes,
            "padding_mask": codec_output.padding_mask,
            "adapted_padding_mask": new_padding_mask,
        }

        return BackfeedingAdapterOutput(**output)


def build_vision_adapter(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_audio_adapter(
    audio_features_sampling_rate: float,
    model_type: Optional[str] = None,
    config_dict: Dict[str, Any] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    parameter_partitioning = (
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with parameter_partitioning:  # Partition the adapter's parameters
        config_dict = config_dict or {}
        model_type_from_config = config_dict.pop("model_type", None)
        model_type = model_type or model_type_from_config
        config = AutoConfig.for_model(model_type, **config_dict)
        common_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
        }
        # TODO(anferico): replace if-else with registry pattern
        if model_type == "mlp":
            audio_adapter = MlpAdapter(**common_kwargs)
        elif model_type == "transformer":
            audio_adapter = TransformerAdapter(
                attn_implementation=attn_implementation, **common_kwargs
            )
        elif model_type == "conv":
            audio_adapter = ConvolutionalAdapter(**common_kwargs)
        elif model_type == "cif":
            audio_adapter = CifAdapter(**common_kwargs)
        elif model_type == "ctc":
            audio_adapter = CtcAdapter(**common_kwargs)
        elif model_type == "cformer":
            audio_adapter = CformerAdapter(
                attn_implementation=attn_implementation, **common_kwargs
            )
        elif model_type == "cmlp":
            audio_adapter = CMlpAdapter(**common_kwargs)
        elif model_type == "qformer":
            audio_adapter = WindowLevelQformerAdapter(
                audio_features_sampling_rate=audio_features_sampling_rate,
                attn_implementation=attn_implementation,
                **common_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown audio adapter model_type: {model_type}."
            )

    print(
        f"Audio adapter [{audio_adapter.__class__.__qualname__}] - "
        f"trainable parameters:"
    )
    print(audio_adapter.get_trainable_parameters(return_plain_dict=False))

    return audio_adapter


def build_backfeeding_adapter(
    audio_features_sampling_rate: float,
    model_type: Optional[str] = None,
    config_dict: Dict[str, Any] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    parameter_partitioning = (
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with parameter_partitioning:  # Partition the adapter's parameters
        config_dict = config_dict or {}
        model_type_from_config = config_dict.pop("model_type", None)
        model_type = model_type or model_type_from_config
        config = AutoConfig.for_model(model_type, **config_dict)
        common_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "audio_features_sampling_rate": audio_features_sampling_rate,
        }
        if model_type == "features":
            backfeeding_adapter = BackfeedingAdapterOnFeatures(**common_kwargs)
        elif model_type == "codes":
            backfeeding_adapter = BackfeedingAdapterOnCodes(**common_kwargs)

        else:
            raise ValueError(
                f"Unknown backfeeding adapter model_type: {model_type}."
            )

        return backfeeding_adapter


def build_video_adapter(
    video_features_sampling_rate: float,
    model_type: Optional[str] = None,
    config_dict: Dict[str, Any] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    # TODO(anferico): of course this function and `build_audio_adapter`
    # should be merged into a single function
    parameter_partitioning = (
        deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with parameter_partitioning:  # Partition the adapter's parameters
        config_dict = config_dict or {}
        model_type_from_config = config_dict.pop("model_type", None)
        model_type = model_type or model_type_from_config
        config = AutoConfig.for_model(model_type, **config_dict)
        common_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
        }

        # TODO(anferico): replace if-else with registry pattern
        if model_type == "mlp":
            video_adapter = MlpAdapter(**common_kwargs)
        elif model_type == "transformer":
            video_adapter = TransformerAdapter(
                attn_implementation=attn_implementation, **common_kwargs
            )
        elif model_type == "conv":
            video_adapter = ConvolutionalAdapter(**common_kwargs)
        elif model_type == "cif":
            video_adapter = CifAdapter(**common_kwargs)
        elif model_type == "ctc":
            video_adapter = CtcAdapter(**common_kwargs)
        elif model_type == "cformer":
            video_adapter = CformerAdapter(
                attn_implementation=attn_implementation, **common_kwargs
            )
        elif model_type == "cmlp":
            video_adapter = CMlpAdapter(**common_kwargs)
        elif model_type == "qformer":
            video_adapter = WindowLevelQformerAdapter(
                # TODO(anferico): give more generic name to the param
                audio_features_sampling_rate=video_features_sampling_rate,
                attn_implementation=attn_implementation,
                **common_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown video adapter model_type: {model_type}."
            )

    print(
        f"Video adapter [{video_adapter.__class__.__qualname__}] - "
        f"trainable parameters:"
    )
    print(video_adapter.get_trainable_parameters(return_plain_dict=False))

    return video_adapter
