import logging
import os

from .model import *  # TODO(anferico): don't do import *
from .model.adapters.configs import (
    AdapterConfig,
    CformerAdapterConfig,
    CifAdapterConfig,
    CmlpAdapterConfig,
    ConvolutionalAdapterConfig,
    CtcAdapterConfig,
    MlpAdapterConfig,
    TransformerAdapterConfig,
    TwoStageAdapterConfig,
    WindowLevelQformerAdapterConfig,
)

logging.basicConfig(level=logging.INFO)

required_env_vars = [
    "DATA_HOME",
    "SPEECHLMM_ROOT",
    "PRETRAINED_COMPONENTS",
    "CHECKPOINTS_HOME",
]
for var in required_env_vars:
    if var not in os.environ:
        raise ValueError(f"Environment variable {var} is not set.")
