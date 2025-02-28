import random

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def get_per_codebook_perturb_threshold(
    alpha: float = 0.0, num_quantizers: int = 8
):
    indices = np.arange(1, num_quantizers + 1)
    scores = 1 / (indices**alpha)
    scores /= np.sum(scores)
    return scores[::-1] * 1.5


class CodeStorm(nn.Module):
    def __init__(
        self,
        alpha: float = 0.8,
        codebook_size: int = 2048,
        num_quantizers: int = 8,
    ):
        super().__init__()
        self.codebook_cardinality = codebook_size
        self.per_codebook_perturb_threshold = (
            get_per_codebook_perturb_threshold(alpha, num_quantizers)
        )
        print(
            f"Storm: alpha: {alpha} codebook size: {codebook_size} num_quantizers: {num_quantizers}"
        )
        print(
            f"Per codebook perturb threshold: {self.per_codebook_perturb_threshold}"
        )

    def forward(self, codes: torch.LongTensor, perturb_prob: float = 0.2):
        # codes are in shape B, T, K
        perturbed_codes = codes.clone()
        if random.random() <= perturb_prob:
            perturbed_codes = rearrange(perturbed_codes, "b t k -> k b t")
            # get mask
            mask = torch.randn_like(perturbed_codes.float())
            mask = [
                cur_mask <= tr
                for cur_mask, tr in zip(
                    mask, self.per_codebook_perturb_threshold
                )
            ]
            mask = torch.stack(mask, dim=0)
            # get perturbations
            perturbations = torch.randint_like(
                perturbed_codes, high=self.codebook_cardinality
            )
            # perturb codes
            perturbed_codes[mask] = perturbations[mask]
            # restore original shape
            perturbed_codes = rearrange(perturbed_codes, "k b t -> b t k")

        return perturbed_codes
