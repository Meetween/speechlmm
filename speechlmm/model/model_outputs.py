from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputWithPastAndGranularLosses(CausalLMOutputWithPast):
    granular_losses: Optional[Dict[str, torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithPastAndGranularLossesAndMetrics(
    CausalLMOutputWithPast
):
    granular_losses: Optional[Dict[str, torch.FloatTensor]] = None
    metrics: Optional[Dict[str, torch.FloatTensor]] = None


@dataclass
class TalkingHeadOutput:
    logits: Optional[torch.FloatTensor] = None
