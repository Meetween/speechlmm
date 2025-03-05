import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import PretrainedConfig
from transformers.modeling_utils import (
    _load_state_dict_into_model,
    load_state_dict,
)

from speechlmm.model.adapters.outputs import SpeechLmmModuleOutput
from speechlmm.model.encoders.video_encoders.auto_avsr_modules.espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder,
)
from speechlmm.model.utils import normalize_model_name_or_path


class AutoAvsrConfig(PretrainedConfig):
    def __init__(
        self,
        _name_or_path=None,
        adim=768,
        aheads=12,
        eunits=3072,
        elayers=12,
        transformer_input_layer="conv3d",
        dropout_rate=0.1,
        transformer_attn_dropout_rate=0.1,
        transformer_encoder_attn_layer_type="rel_mha",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=31,
        zero_triu=False,
        a_upsample_ratio=1,
        relu_type="swish",
        ddim=768,
        dheads=12,
        dunits=3072,
        dlayers=6,
        lsm_weight=0.1,
        transformer_length_normalized_loss=False,
        mtlalpha=0.1,
        ctc_type="builtin",
        rel_pos_type="latest",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name_or_path = _name_or_path
        self.adim = adim
        self.aheads = aheads
        self.eunits = eunits
        self.elayers = elayers
        self.transformer_input_layer = transformer_input_layer
        self.dropout_rate = dropout_rate
        self.transformer_attn_dropout_rate = transformer_attn_dropout_rate
        self.transformer_encoder_attn_layer_type = (
            transformer_encoder_attn_layer_type
        )
        self.macaron_style = macaron_style
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.zero_triu = zero_triu
        self.a_upsample_ratio = a_upsample_ratio
        self.relu_type = relu_type
        self.ddim = ddim
        self.dheads = dheads
        self.dunits = dunits
        self.dlayers = dlayers
        self.lsm_weight = lsm_weight
        self.transformer_length_normalized_loss = (
            transformer_length_normalized_loss
        )
        self.mtlalpha = mtlalpha
        self.ctc_type = ctc_type
        self.rel_pos_type = rel_pos_type

    # def to_dict(self):
    #     return deepcopy(self.__dict__)


# TODO(anferico): create a base VideoEncoder class and have this extend that
class AutoAvsrEncoder(nn.Module):
    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        delay_load: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if config_dict is None or name_or_path is None:
            raise NotImplementedError

        self.name_or_path = normalize_model_name_or_path(
            name_or_path,
            allow_hf_hub=False,
            # ↑ NOTE: Auto-AVSR is not available on Hugging Face Hub yet
        )
        self.torch_dtype = torch_dtype
        self.device = device

        self.config = AutoAvsrConfig(**config_dict)

        self.is_loaded = False
        if not delay_load:
            self._load_model()

    def _load_model(self):
        self.model = Encoder(
            attention_dim=self.config.adim,
            attention_heads=self.config.aheads,
            linear_units=self.config.eunits,
            num_blocks=self.config.elayers,
            input_layer=self.config.transformer_input_layer,
            dropout_rate=self.config.dropout_rate,
            positional_dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.transformer_attn_dropout_rate,
            encoder_attn_layer_type=self.config.transformer_encoder_attn_layer_type,
            macaron_style=self.config.macaron_style,
            use_cnn_module=self.config.use_cnn_module,
            cnn_module_kernel=self.config.cnn_module_kernel,
            zero_triu=getattr(self.config, "zero_triu", False),
            a_upsample_ratio=self.config.a_upsample_ratio,
            relu_type=getattr(self.config, "relu_type", "swish"),
        )

        logging.info("Loading AUTO_AVSR weights")
        state_dict = load_state_dict(self.name_or_path)
        state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        consume_prefix_in_state_dict_if_present(state_dict, "encoder.")
        _load_state_dict_into_model(self.model, state_dict, start_prefix="")

        self.model.to(device=self.device)
        self.is_loaded = True

    def forward(self, videos_srs, *args, **kwargs):
        # TODO(anferico): the way video tensors and attention masks are
        # created is quite weird and should be refactored
        max_len = max([v.shape[0] for v, sr in videos_srs])
        v0, _ = videos_srs[0]
        vid_tensor = v0.new_zeros((len(videos_srs), max_len, 1, 88, 88))
        mask_ = v0.new_ones((len(videos_srs), max_len, max_len))
        return_tensor = torch.zeros(
            vid_tensor.shape[0],
            vid_tensor.shape[1],
            768,
            dtype=vid_tensor.dtype,
            device=vid_tensor.device,
        )
        # TODO(anferico): do batch processing instead of for loop
        for id_, (elem, _) in enumerate(videos_srs):
            vid_tensor[id_, : elem.shape[0]] = elem
            mask_[id_, : elem.shape[0], : elem.shape[0]] = 0
            videos = self.model(elem.unsqueeze(0), None)
            return_tensor[id_, : elem.shape[0]] = videos[0][0]

        mask_ = mask_.to(torch.bool)
        videos = return_tensor.to(self.torch_dtype)
        return SpeechLmmModuleOutput(
            features=videos, padding_mask=mask_[:, :, 0]
        )

    def forward_batch(self, x, *args, **kwargs):
        max_len = max([v.shape[0] for v in x])
        vid_tensor = x[0].new_zeros((len(x), max_len, 1, 88, 88))
        mask_ = x[0].new_ones((len(x), max_len, max_len))
        for id_, elem in enumerate(x):
            vid_tensor[id_, : elem.shape[0]] = elem
            mask_[id_, : elem.shape[0], : elem.shape[0]] = 0
        mask_ = mask_.to(torch.bool)
        videos = self.model(vid_tensor, ~mask_)
        videos = videos[0]
        return SpeechLmmModuleOutput(
            features=videos, padding_mask=mask_[:, :, 0]
        )

    @property
    def input_sampling_rate(self):
        return 25  # TODO(anferico): do NOT hardcode, get from config
