import contextlib
import math
from functools import singledispatch
import os
from pathlib import Path

import torch
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError


@singledispatch
def compute_output_length_from_conv1d_hyperparams(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return math.floor(
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    )


@compute_output_length_from_conv1d_hyperparams.register
def _(
    input_length: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    return torch.floor(
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1)
        / stride
        + 1
    ).long()


@singledispatch
def compute_output_length_from_conv1d_layer(
    input_length: int, conv1d_layer: torch.nn.Conv1d
) -> int:
    return compute_output_length_from_conv1d_hyperparams(
        input_length,
        conv1d_layer.kernel_size[0],
        conv1d_layer.stride[0],
        conv1d_layer.padding[0],
        conv1d_layer.dilation[0],
    )


@compute_output_length_from_conv1d_layer.register
def _(input_length: torch.Tensor, conv1d_layer: torch.nn.Conv1d):
    return compute_output_length_from_conv1d_hyperparams(
        input_length,
        conv1d_layer.kernel_size[0],
        conv1d_layer.stride[0],
        conv1d_layer.padding[0],
        conv1d_layer.dilation[0],
    )


def get_candidate_modules_to_save_for_lora(
    module: torch.nn.Module, prefix: str = ""
):
    """
    Get all candidate `modules_to_save` for LoRA, that is, modules that should be saved alongside
    LoRA adapters in the checkpoint because they might contain one or more trainable parameters.
    A module is a candidate if:
    - it is a "leaf" module (i.e. it does not have any submodules)
    - it has "loose" trainable parameters (i.e. parameters that are not bound to any submodules)

    Rationale: as the name suggests, `modules_to_save` is supposed to contain modules (i.e.
    `torch.nn.Module` instances), not parameters. If the module's a leaf module, then it's all good.
    However, if the module is NOT a leaf module (has children submodules) and has trainable
    parameters as its immediate children, then the best we can do is return the module itself.
    Note that if the module has frozen parameters in any of its submodules, this will cause such
    parameters to be included in the LoRA checkpoint even though it wouldn't be necessary. Still,
    we have to live with this for now (cf. https://github.com/huggingface/peft/discussions/2217).
    """
    if any(p.requires_grad for p in module.parameters(recurse=False)):
        yield (prefix, module)
    else:
        is_leaf = True
        for child_name, child in module.named_children():
            is_leaf = False
            qualified_child_name = (
                f"{prefix}.{child_name}" if len(prefix) > 0 else child_name
            )
            yield from get_candidate_modules_to_save_for_lora(
                child, prefix=qualified_child_name
            )
        if is_leaf:
            yield (prefix, module)

def normalize_model_name_or_path(
    model_name_or_path: str, allow_hf_hub: bool = True
) -> str:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return str(model_path)

    pretrained_components_dir = os.getenv("PRETRAINED_COMPONENTS", None)
    if pretrained_components_dir is not None:
        model_path = Path(pretrained_components_dir, model_name_or_path)
        if model_path.exists():
            return str(model_path)

    error_msg = (
        f"Model '{model_name_or_path}' could not be found. Please make "
        f"sure the model exists either:\n"
        f"  - locally at {model_name_or_path}\n"
        f"  - locally at $PRETRAINED_COMPONENTS/{model_name_or_path} (in "
        f"which case you must set the environment variable "
        f"$PRETRAINED_COMPONENTS to a suitable path)"
    )

    if allow_hf_hub:
        with contextlib.suppress(RepositoryNotFoundError):
            _ = model_info(model_name_or_path)
            return model_name_or_path
        error_msg += "\n  - in the Hugging Face Hub"

    raise ValueError(error_msg)
