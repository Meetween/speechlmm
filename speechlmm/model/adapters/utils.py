import json
from collections import UserDict
from typing import Optional

import torch

from speechlmm.model.utils import compute_output_length_from_conv1d_layer


def count_parameters(
    module: torch.nn.Module, trainable_only: bool = False
) -> int:
    # if the module parameters have been partitioned into multiple GPUs
    # by DeepSpeed ZeRO-3, we must get the number of elements of each
    # parameter from the `ds_numel` attribute
    return sum(
        getattr(param, "ds_numel", param.numel())
        for param in module.parameters()
        if not trainable_only or param.requires_grad
    )


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return count_parameters(module, trainable_only=True)


def lengths_to_attention_mask(lengths: torch.LongTensor) -> torch.BoolTensor:
    return (
        torch.arange(lengths.max(), device=lengths.device)[None, :]
        < lengths[:, None]
    )


def lengths_to_padding_mask(lengths: torch.LongTensor) -> torch.BoolTensor:
    return ~lengths_to_attention_mask(lengths)


@torch.no_grad()
def attention_weights_to_attention_mask(
    attention_weights_per_layer: Optional[torch.FloatTensor] = None,
    # ↑ `num_layers` tensors of shape (batch_size, num_heads, seq_len, seq_len)
) -> torch.BoolTensor:
    # filter out attention weights that are None (due to layer dropout)
    attention_weights_per_layer = tuple(
        aw for aw in attention_weights_per_layer if aw is not None
    )

    ATTENTION_HEADS_DIM = 1
    total_attention_weights = sum(attention_weights_per_layer).sum(
        dim=ATTENTION_HEADS_DIM
    )
    # ↑ attention weights might be exactly zero even for non-padding
    # tokens in some layers and/or heads within a layer, but it's very
    # unlikely that they are for ALL layers and ALL heads within a layer
    ANY_ONE_DIM = 0
    attention_mask = total_attention_weights[:, ANY_ONE_DIM] > 0

    # for some reason, in some cases, the attention mask for the longest
    # sample in the batch is not all ones, but has a zero at the end
    # (and presumably the same holds for the other samples in the batch).
    # While this is not a big deal generally speaking, it causes the
    # `ConvolutionalAdapter` to crash as for that projector, such
    # assumption is crucial when building the attention/padding mask of
    # the output after the convolutional layers. To fix this, we simply
    # add one to the length of all samples in the batch
    SEQ_LEN_DIM = 1
    lengths = attention_mask.sum(dim=SEQ_LEN_DIM)  # (batch_size,)
    if lengths.max() < attention_mask.shape[SEQ_LEN_DIM]:
        attention_mask = lengths_to_attention_mask(lengths + 1)

    return attention_mask


class TotalTrackingDict(UserDict):
    def __init__(self, *args, **kwargs):
        self.total = 0
        super().__init__(*args, **kwargs)
        # If `TotalTrackingDict` is instantiated with a dictionary
        # having a "total" key (e.g. the result of calling `to_dict` on
        # another `TotalTrackingDict` instance), the `.total` attribute
        # gets set to twice the actual total. For this reason, we remove
        # the "total" key from the dictionary to trigger a recalculation
        # of the `.total` attribute (see `__delitem__` below)
        self.pop("total", None)

    def _update_total_add(self, value):
        if isinstance(value, int):
            self.total += value
        elif isinstance(value, list):
            self.total += sum(value)
        elif isinstance(value, TotalTrackingDict):
            self.total += value.total

    def _update_total_remove(self, value):
        if isinstance(value, int):
            self.total -= value
        elif isinstance(value, list):
            self.total -= sum(value)
        elif isinstance(value, TotalTrackingDict):
            self.total -= value.total

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("Keys must be strings.")
        if not isinstance(value, (int, list, TotalTrackingDict)):
            raise ValueError(
                "Value must be an int, list, or `TotalTrackingDict` instance."
            )

        # Adjust the total by removing the current value associated with
        # the key (if any)
        if key in self:
            self._update_total_remove(self[key])

        # Set the new value and update the total
        super().__setitem__(key, value)
        self._update_total_add(value)

    def __delitem__(self, key):
        if key in self:
            # Adjust the total by removing the value associated with the
            # key being removed
            self._update_total_remove(self[key])
            super().__delitem__(key)
        else:
            raise KeyError(f"Key '{key}' not found.")

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, TotalTrackingDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        result["total"] = self.total
        return result


class PaddingAwareConv1d(torch.nn.Conv1d):
    """Conv1d layer that resets padded positions to 0 after convolving over the (padded) input.
    Fixes 🪲2 described here: https://arxiv.org/pdf/2303.16166
    """

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.BoolTensor
    ) -> torch.Tensor:
        # inputs: (batch_size, in_channels, seq_len)
        # padding_mask: (batch_size, seq_len)
        # outputs: (batch_size, out_channels, seq_len)
        convolved_inputs = super().forward(inputs)
        convolved_padding_mask = self._convolve_padding_mask(padding_mask).to(
            device=padding_mask.device
        )
        return (
            convolved_inputs.masked_fill(
                mask=convolved_padding_mask.unsqueeze(1), value=0
            ),
            convolved_padding_mask,
        )

    @torch.no_grad()
    def _convolve_padding_mask(self, padding_mask: torch.BoolTensor):
        def assert_is_symmetric(param_name: str):
            param = getattr(self, param_name)
            if not 0 < len(param) < 3:
                raise ValueError(
                    f"'{param_name}' must be a tuple of length 1 or 2."
                )
            if len(param) == 2 and param[0] != param[1]:
                raise ValueError(
                    f"'{param_name}' must be symmetric (got {param})."
                )

        for param_name in ["padding", "kernel_size", "stride", "dilation"]:
            # This allows us to safely do parameter[0] below
            assert_is_symmetric(param_name)

        SEQ_LEN_DIM = 1
        pre_conv_lengths = (~padding_mask).sum(SEQ_LEN_DIM)
        post_conv_lengths = compute_output_length_from_conv1d_layer(
            pre_conv_lengths, conv1d_layer=self
        )
        return lengths_to_padding_mask(post_conv_lengths.long())


class Interpolation(torch.nn.Module):
    r"""
    See :func:`torch.nn.functional.interpolate` for more information about the
    arguments (this is just a wrapper).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.nn.functional.interpolate(x, *self.args, **self.kwargs)
