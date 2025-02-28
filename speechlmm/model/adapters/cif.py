"""
Adapted from https://github.com/George0828Zhang/torch_cif/blob/f520e1e437f587f4c1cbd5063b4d4c726f6a76be/torch_cif/cif.py
"""

from typing import Dict, Optional

import torch


class Cif(torch.nn.Module):
    r"""A fast parallel implementation of continuous integrate-and-fire (CIF)
    Reference: https://arxiv.org/abs/1905.11235
    """

    def __init__(
        self,
        firing_threshold: float = 1.0,
        residual_threshold: float = 0.5,
        unbound_input_weights: bool = False,
        eps: float = 1e-4,
    ):
        r"""
        Args:
            firing_threshold (float):
              The threshold used to determine when to fire.
            residual_threshold (float):
              The threshold for determining firing for tail handling.
            unbound_input_weights (bool, optional):
              Whether to check if 0 <= input_weights <= 1.
            eps (float, optional):
              ε to prevent underflow for divisions. Default: 1e-4
        """
        super().__init__()
        self.firing_threshold = firing_threshold
        self.residual_threshold = residual_threshold
        self.unbound_input_weights = unbound_input_weights
        self.eps = eps

    def forward(
        self,
        inputs: torch.FloatTensor,  # audio features
        input_weights: torch.FloatTensor,  # weights corresponding to each element in the inputs
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # padding mask of the audio features
        target_lengths: Optional[
            torch.LongTensor
        ] = None,  # desired length of the targets for each sample in the minibatch
    ) -> Dict[str, torch.Tensor]:
        r"""
        Shapes:
            B: batch size
            S: source (encoder) sequence length
            C: source feature dimension
            T: target sequence length

        Args:
            inputs (torch.Tensor):
              Input features to be integrated. Shape: (B, S, C)
            input_weights (torch.FloatTensor):
              Weights corresponding to each elements in the inputs. It is expected to be after sigmoid function. Shape: (B, S)
            padding_mask (torch.Tensor, optional):
              A binary mask representing padded elements in the inputs. 1 is padding, 0 is not. Shape: (B, S)
            target_lengths (torch.Tensor, optional):
              Desired length of the targets for each sample in the minibatch. Shape: (B,)

        Returns -> Dict[str, List[torch.Tensor]]: Key/values described below.
            cif_out:
              The output integrated from the source. Shape: (B, T, C)
            cif_lengths:
              The output length for each element in batch. Shape: (B,)
            input_weights_sum:
              The sum of input_weights for each element in batch. Can be used to compute the quantity loss. Shape: (B,)
            delays:
              The expected delay (in terms of source tokens) for each target tokens in the batch. Shape: (B, T)
            tail_weights:
              During inference, return the tail. Shape: (B,)
            scaled_alpha:
              input_weights after applying weight scaling. Shape: (B, S)
            cumsum_alpha:
              cumsum of input_weights after scaling. Shape: (B, S)
            right_indices:
              right scatter indices, or floor(cumsum(input_weights)). Shape: (B, S)
            right_weights:
              right scatter weights. Shape: (B, S)
            left_indices:
              left scatter indices. Shape: (B, S)
            left_weights:
              left scatter weights. Shape: (B, S)
        """

        B, S, C = inputs.size()
        assert tuple(input_weights.size()) == (
            B,
            S,
        ), f"{input_weights.size()} != {(B, S)}"
        assert not torch.isnan(
            input_weights
        ).any(), "NaN in input_weights tensor."
        if not self.unbound_input_weights:
            assert (
                input_weights.le(1.0 + self.eps).all()
                and input_weights.ge(0.0 - self.eps).all()
            ), (
                "Incorrect values in input_weights tensor. Expected "
                "0.0 <= input_weights <= 1.0"
            )
            input_weights = input_weights.clip(min=0.0, max=1.0)

        if padding_mask is not None:
            assert not padding_mask[
                :, 0
            ].any(), "Expected right-padded inputs."
            input_weights = input_weights.masked_fill(padding_mask.bool(), 0.0)

        input_weights_sum = input_weights.sum(-1)
        at_training_time = target_lengths is not None
        if at_training_time:
            assert tuple(target_lengths.size()) == (
                B,
            ), f"{tuple(target_lengths.size())} != {(B,)}"
            # apply scaling strategy described in §3.2 of the CIF paper
            desired_sum = (
                self.firing_threshold * target_lengths.type_as(inputs)
            ) + self.eps
            input_weights = input_weights * (
                desired_sum / input_weights_sum
            ).unsqueeze(1)
        else:  # inference mode
            target_lengths = (
                (input_weights_sum / self.firing_threshold).floor().long()
            )

        T = target_lengths.max()

        # aggregate and integrate
        input_weights_cumulative_sum = input_weights.cumsum(
            -1, dtype=torch.float64
        )
        with torch.no_grad():
            # indices used for scattering
            right_idx = (
                (input_weights_cumulative_sum / self.firing_threshold)
                .floor()
                .long()
                .clip(min=0, max=T)
            )
            left_idx = right_idx.roll(1, dims=1)
            left_idx[:, 0] = 0

            # count number of firings from each source
            n_firings = right_idx - left_idx
            extra_weights = (n_firings - 1).clip(min=0)

        # The extra entry in last dim is for tail
        output = inputs.new_zeros((B, T + 1, C))
        delay = inputs.new_zeros((B, T + 1))
        source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(inputs)
        zero = input_weights.new_zeros((1,))

        # right scatter
        right_weight = torch.where(
            n_firings > 0,
            (
                input_weights_cumulative_sum
                - (right_idx.type_as(input_weights) * self.firing_threshold)
            ),
            zero,
        ).type_as(inputs)
        output.scatter_add_(
            dim=1,
            index=right_idx.unsqueeze(-1).expand(-1, -1, C),
            src=right_weight.unsqueeze(-1) * inputs,
        )
        delay.scatter_add_(
            dim=1,
            index=right_idx,
            src=right_weight * source_range / self.firing_threshold,
        )

        # left scatter
        left_weight = (
            input_weights
            - right_weight
            - (extra_weights.type_as(input_weights) * self.firing_threshold)
        ).type_as(inputs)
        output.scatter_add_(
            dim=1,
            index=left_idx.unsqueeze(-1).expand(-1, -1, C),
            src=left_weight.unsqueeze(-1) * inputs,
        )
        delay.scatter_add_(
            dim=1,
            index=left_idx,
            src=left_weight * source_range / self.firing_threshold,
        )

        # extra scatters
        if extra_weights.ge(0).any():
            extra_steps = extra_weights.max().item()
            tgt_idx = left_idx
            src_feats = inputs * self.firing_threshold
            for _ in range(extra_steps):
                tgt_idx = (tgt_idx + 1).clip(max=T)
                # (B, S, 1)
                src_mask = extra_weights > 0
                output.scatter_add_(
                    dim=1,
                    index=tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                    src=src_feats * src_mask.unsqueeze(2),
                )
                delay.scatter_add_(
                    dim=1, index=tgt_idx, src=source_range * src_mask
                )
                extra_weights -= 1

        # tail handling
        if at_training_time:
            # ignore tail
            output = output[:, :T, :]
            delay = delay[:, :T]
        else:
            # find out contribution to output tail
            # note: w/o scaling, extra weight is all 0
            zero = right_weight.new_zeros((1,))
            r_mask = right_idx == target_lengths.unsqueeze(1)
            tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
            l_mask = left_idx == target_lengths.unsqueeze(1)
            tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

            # a size (B,) mask that extends position that passed threshold.
            extend_mask = tail_weights >= self.residual_threshold

            # extend 1 fire and upscale the weights
            if extend_mask.any():
                # (B, T, C), may have infs so need the mask
                upscale = (
                    torch.ones_like(output)
                    .scatter(
                        dim=1,
                        index=target_lengths.view(B, 1, 1).expand(-1, -1, C),
                        src=self.firing_threshold
                        / (
                            tail_weights.masked_fill(
                                ~extend_mask,
                                self.firing_threshold,
                            )
                            .view(B, 1, 1)
                            .expand(-1, -1, C)
                        ),
                    )
                    .detach()
                )
                output *= upscale
                target_lengths += extend_mask.long()
                T = target_lengths.max()
            output = output[:, :T, :]
            delay = delay[:, :T]

            # a size (B, T) mask to erase weights
            tail_mask = torch.arange(T, device=output.device).unsqueeze(
                0
            ) >= target_lengths.unsqueeze(1)
            output[tail_mask] = 0

        # calculate durations
        durations = []
        for padded_idx, target_length, audio_length in zip(
            input_weights.cumsum(-1, dtype=torch.float64).round().long(),
            target_lengths,
            (~padding_mask).sum(-1),
        ):
            idx = padded_idx[padded_idx < target_length]
            idx = torch.cat([idx, target_length.unsqueeze(0)])
            duration = torch.where(torch.diff(idx))[0] + 1
            duration = torch.cat(
                [duration[0].unsqueeze(0), torch.diff(duration)]
            )
            if duration.size(-1) < target_length:
                duration = torch.cat(
                    [duration, torch.zeros(1).to(duration.device)]
                )

            # insert in the last duration the tail
            duration[-1] += audio_length - duration.sum()
            durations += [duration]

        durations = torch.nn.utils.rnn.pad_sequence(
            durations, batch_first=True
        ).long()

        return {
            "integrated_embeddings": output,
            "integrated_embeddings_lengths": target_lengths,
            "input_weights_sum": input_weights_sum,
            "delays": delay,
            "tail_weights": tail_weights if target_lengths is None else None,
            "scaled_input_weights": input_weights,
            "input_weights_cumulative_sum": input_weights_cumulative_sum,
            "right_indices": right_idx,
            "right_weights": right_weight,
            "left_indices": left_idx,
            "left_weights": left_weight,
            "durations": durations,
        }
