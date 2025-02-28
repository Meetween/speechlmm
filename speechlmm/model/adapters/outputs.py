from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SpeechLmmModuleOutput:
    features: torch.FloatTensor
    padding_mask: torch.BoolTensor


@dataclass
class CodecOutput(SpeechLmmModuleOutput):
    codes: torch.LongTensor
    audio_features_per_codebook: Optional[torch.FloatTensor] = None


@dataclass
class WindowQformerOutput(SpeechLmmModuleOutput):
    windowed_cross_attention_mask: Optional[torch.BoolTensor] = None
    durations: Optional[torch.LongTensor] = None
    triplet_loss: Optional[torch.FloatTensor] = None


@dataclass
class BackfeedingAdapterOutput(SpeechLmmModuleOutput):
    adapted_padding_mask: torch.BoolTensor
    codes: torch.LongTensor
    durations: Optional[torch.LongTensor] = None
    audio_features_per_codebook: Optional[torch.FloatTensor] = None
    windowed_cross_attention_mask: Optional[torch.BoolTensor] = None
    # ⬆ NOTE: the durations are setted in the backfeeding process


@dataclass
class CifAdapterOutput(SpeechLmmModuleOutput):
    ctc_loss: Optional[torch.FloatTensor] = None
    quantity_loss: Optional[torch.FloatTensor] = None


@dataclass
class CtcAdapterOutput(SpeechLmmModuleOutput):
    ctc_loss: Optional[torch.FloatTensor] = None
