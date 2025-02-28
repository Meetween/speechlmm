import logging
import math
from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch

from speechlmm.model.attn_implementation import AttentionImplementationMixin
from speechlmm.model.model_outputs import TalkingHeadOutput
from speechlmm.model.sampling import sample_token

logger = logging.getLogger(__name__)

from functools import partial

from torch import nn
from transformers import AutoConfig, BertConfig, BertModel, PretrainedConfig

from speechlmm.constants import IGNORE_INDEX
from speechlmm.model.adapters.outputs import CodecOutput

# from speechlmm.model.adapters.qformer import BertLMHeadModel
from speechlmm.model.adapters.qformer import BertConfig as QformerConfig
from speechlmm.model.adapters.qformer import BertModel as QformerModel
from speechlmm.model.embeddings import ScaledEmbedding


class TalkingHeadConfig(PretrainedConfig):
    def __init__(
        self,
        num_quantizers: Optional[int] = None,
        codebook_size: Optional[int] = None,
        **kwargs,
    ):
        """
        num_quantizers: int - Number of quantized tokens
        codebook_size: int - Size of the codebook
        text_vocab_size: int - Size of the LLM vocabulary
        """
        super().__init__(**kwargs)
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size


class MoshiTalkingHeadConfig(TalkingHeadConfig):
    model_type = "moshi_bert"

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 2048,
        text_vocab_size: int = 32004,
        tempformer_dim: int = 4096,
        depformer_dim: int = 1024,
        depformer_dim_feedforward: int = int(4.125 * 1024),
        depformer_num_heads: int = 16,
        depformer_num_layers: int = 6,
        depformer_casual: bool = True,
        max_position_embeddings: int = 4096,
        use_text_tokens: bool = True,
        **kwargs,
    ):
        """
        tempformer_dim: int - Hidden dimension of the Temporal Transformer
        depformer_dim: int - Hidden dimension of the Depth Transformer
        depformer_dim_feedforward: int - Feedforward dimension of the Depth Transformer
        depformer_num_heads: int - Number of heads in the Depth Transformer
        depformer_num_layers: int - Number of layers in the Depth Transformer
        depformer_casual: bool - Whether the Depth Transformer is causal
        """

        super().__init__(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            **kwargs,
        )
        self.text_vocab_size = text_vocab_size
        self.tempformer_dim = tempformer_dim
        self.depformer_dim = depformer_dim
        self.depformer_dim_feedforward = depformer_dim_feedforward
        self.depformer_num_heads = depformer_num_heads
        self.depformer_num_layers = depformer_num_layers
        self.depformer_casual = depformer_casual
        self.max_position_embeddings = max_position_embeddings
        self.use_text_tokens = use_text_tokens


class NARTalkingHeadConfig(TalkingHeadConfig):
    model_type = "qformer_talking_head"

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 2048,
        tempformer_dim: int = 4096,
        depformer_casual: bool = True,
        num_query_tokens: int = 8,
        duration_prediction: bool = False,
        depformer_multiple_heads: bool = False,
        depformer_inner_batch_size: int = 1024,
        depformer_expansion_factor: Optional[int] = 1,
        depformer_add_text_tokens: bool = False,
        text_vocab_size: Optional[int] = None,
        cross_attention_window_size: int = 1,
        **kwargs,
    ):
        """
        tempformer_dim: int - Hidden dimension of the Temporal Transformer
        depformer_dim: int - Hidden dimension of the Depth Transformer
        depformer_dim_feedforward: int - Feedforward dimension of the Depth Transformer
        depformer_num_heads: int - Number of heads in the Depth Transformer
        depformer_num_layers: int - Number of layers in the Depth Transformer
        depformer_casual: bool - Whether the Depth Transformer is causal
        """

        super().__init__(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            **kwargs,
        )
        self.tempformer_dim = tempformer_dim
        self.depformer_casual = depformer_casual
        self.depformer_multiple_heads = depformer_multiple_heads

        # qformer specific
        self.num_query_tokens = num_query_tokens
        self.duration_prediction = duration_prediction
        self.depformer_add_text_tokens = depformer_add_text_tokens
        self.depformer_inner_batch_size = depformer_inner_batch_size
        self.depformer_expansion_factor = depformer_expansion_factor
        self.depformer_text_vocab_size = text_vocab_size
        self.cross_attention_window_size = cross_attention_window_size


AutoConfig.register(model_type="moshi_bert", config=MoshiTalkingHeadConfig)
AutoConfig.register(
    model_type="qformer_talking_head", config=NARTalkingHeadConfig
)


class TalkingHead(
    AttentionImplementationMixin, torch.nn.Module, metaclass=ABCMeta
):
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(
        self,
        config: TalkingHeadConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.set_attn_implementation_with_fallback(attn_implementation)

        self.config = config
        self.torch_dtype = torch_dtype
        metrics = []
        for k in range(self.config.num_quantizers):
            metrics.append(f"accuracy_{k}")
            metrics.append(f"top10_accuracy_{k}")
        metrics.append("accuracy")
        metrics.append("top10_accuracy")
        self.metrics = metrics

    @property
    def dtype(self):
        return self.torch_dtype


class MoshiBertTalkingHead(TalkingHead):
    _supports_flash_attn_2 = BertModel._supports_flash_attn_2
    _supports_sdpa = BertModel._supports_sdpa

    def __init__(
        self,
        config: MoshiTalkingHeadConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        self.tempformer_dim = self.config.tempformer_dim
        self.depformer_dim = self.config.depformer_dim
        self.depformer_dim_feedforward = self.config.depformer_dim_feedforward
        self.depformer_num_heads = self.config.depformer_num_heads
        self.depformer_num_layers = self.config.depformer_num_layers
        self.depformer_casual = self.config.depformer_casual
        self.max_position_embeddings = self.config.max_position_embeddings
        self.use_text_tokens = self.config.use_text_tokens

        self.norm = "rms_norm"  # FIXME hardcoded
        self.text_vocab_size = self.config.text_vocab_size
        print(
            f"{self.__class__.__name__} initialized with text_vocab_size: {self.text_vocab_size}"
        )

        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=self.norm,
            zero_idx=IGNORE_INDEX,
            dtype=self.torch_dtype,
        )

        # Kwargs
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size

        # Linear layers for depformer_in
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(
                    self.tempformer_dim,
                    self.depformer_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.num_quantizers)
            ]
        )

        # Embeddings for depformer_emb
        self.depformer_emb = nn.ModuleList(
            [
                EmbeddingFactory(self.codebook_size + 1, self.depformer_dim)
                for _ in range(self.num_quantizers - 1)
            ]
        )
        if self.use_text_tokens:
            self.depformer_text_emb = EmbeddingFactory(
                self.text_vocab_size, self.depformer_dim
            )

        # Linear layers for output (linears)
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    self.depformer_dim,
                    self.codebook_size,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.num_quantizers)
            ]
        )

        # Bert Transformer initialization
        config = BertConfig(
            hidden_size=self.depformer_dim,
            num_attention_heads=self.depformer_num_heads,
            intermediate_size=self.depformer_dim_feedforward,
            num_hidden_layers=self.depformer_num_layers,
            is_decoder=self.depformer_casual,  # Adjust for causal architecture
            max_position_embeddings=self.max_position_embeddings,
            torch_dtype=self.torch_dtype,
        )
        self.dep_transformer = BertModel(config).to(dtype=self.torch_dtype)

    def forward(
        self, context_vector, context_attention_mask, text_tokens, codes
    ):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            context_attention_mask: [B, S]
            text_tokens: [B, S]
            codes: [B, S, K]
        """
        B, S = context_vector.shape[:2]

        # NOTE: the code below does not handle context_attention_mask
        if self.use_text_tokens:
            sequence = torch.cat(
                [text_tokens.unsqueeze(-1), codes[:, :, :-1]], dim=-1
            )  # B, S, K
        else:
            sequence = torch.cat(
                [torch.zeros_like(codes[:, :, :1]), codes], dim=-1
            )

        x = context_vector

        # pass through depformer_in linear layers
        depformer_inputs = [
            self.depformer_in[i](x) for i in range(len(self.depformer_in))
        ]

        # process with depformer_emb embeddings
        last_token_input = []
        text_emb = None
        for codebook_idx in range(self.num_quantizers):
            if codebook_idx == 0:
                if self.use_text_tokens:
                    text_emb = self.depformer_text_emb(sequence[:, :, 0])
                else:
                    text_emb = (
                        sequence[:, :, 0]
                        .unsqueeze(-1)
                        .repeat(1, 1, self.depformer_dim)
                    )
                last_token_input.append(text_emb)
            else:
                last_token_input.append(
                    self.depformer_emb[codebook_idx - 1](
                        sequence[:, :, codebook_idx]
                    )
                )

        last_token_input = torch.stack(
            last_token_input, dim=1
        )  # B, K, S, depformer_dim

        depformer_inputs = [
            depformer_inputs[i] + last_token_input[:, i, :]
            for i in range(len(depformer_inputs))
        ]

        inputs_embeds = torch.stack(
            depformer_inputs, dim=1
        )  # B, K, S, depformer_dim
        inputs_embeds = inputs_embeds.permute(
            0, 2, 1, 3
        )  # B, S, K, depformer_dim
        B, S, K, depformer_dim = inputs_embeds.shape
        dep_outputs = torch.empty(
            B,
            S,
            K,
            depformer_dim,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        # flatten the inputs_embeds to B, S*K, depformer_dim
        if B * S <= self.max_position_embeddings:
            inputs_embeds_flat = inputs_embeds.reshape(B * S, K, depformer_dim)
            # Process all time steps at once if the transformer can handle (B * S, K, depformer_dim) inputs
            outputs_flat = self.dep_transformer(
                inputs_embeds=inputs_embeds_flat
            ).last_hidden_state
            # Reshape it back to original shape
            dep_outputs = outputs_flat.reshape(B, S, K, depformer_dim)
        else:
            for b in range(B):
                outputs = self.dep_transformer(
                    inputs_embeds=inputs_embeds[b, :, :, :]
                ).last_hidden_state
                dep_outputs[b, :, :, :] = outputs

        # pass through linears to get logits
        logits = [
            self.linears[i](dep_outputs[:, :, i, :])
            for i in range(self.num_quantizers)
        ]
        logits = torch.stack(logits, dim=2)  # B, S, K, codebook_size

        return TalkingHeadOutput(
            logits=logits,
        )

    @torch.no_grad()
    def generate(
        self,
        context_vector,
        text_token,
        sample=False,
        temperature=1.0,
        top_k=0,
        semantic_only=False,
    ):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            text_token: [B, S]
        """
        B, S = context_vector.shape[:2]
        assert B == 1
        assert S == 1

        sequence = text_token.unsqueeze(-1)

        for q in range(self.num_quantizers):
            # print(f"context_vector: {context_vector}")
            # print(f"sequence: {sequence}")
            logits = self.generation_step(context_vector, sequence)[
                :, :, -1, :
            ]
            next_token = sample_token(
                logits.float(),
                use_sampling=False,
                temp=temperature,
                top_k=top_k,
            )
            if semantic_only:
                return torch.cat(
                    [
                        next_token.unsqueeze(-1),
                        torch.full(
                            (1, 1, self.num_quantizers - 1),
                            IGNORE_INDEX,
                            dtype=next_token.dtype,
                            device=next_token.device,
                        ),
                    ],
                    dim=-1,
                )
            sequence = torch.cat([sequence, next_token.unsqueeze(-1)], dim=-1)
        codes = sequence[:, :, 1:]
        return codes

    def generation_step(self, context_vector, sequence):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            sequence: [B, S, K]
        """
        B, S, K = sequence.shape
        assert B == 1
        assert S == 1
        assert K <= self.num_quantizers

        x = context_vector
        cur_num_quantizers = sequence.shape[-1]

        # pass through depformer_in linear layers
        depformer_inputs = [
            self.depformer_in[i](x) for i in range(cur_num_quantizers)
        ]

        # process with depformer_emb embeddings
        last_token_input = []
        for codebook_idx in range(cur_num_quantizers):
            if codebook_idx == 0:
                text_emb = self.depformer_text_emb(sequence[:, :, 0])
                last_token_input.append(text_emb)
            else:
                last_token_input.append(
                    self.depformer_emb[codebook_idx - 1](
                        sequence[:, :, codebook_idx]
                    )
                )

        last_token_input = torch.stack(
            last_token_input, dim=1
        )  # B, K, S, depformer_dim
        depformer_inputs = [
            depformer_inputs[i] + last_token_input[:, i, :]
            for i in range(cur_num_quantizers)
        ]

        inputs_embeds = torch.stack(
            depformer_inputs, dim=1
        )  # B, K, S, depformer_dim
        inputs_embeds = inputs_embeds.permute(
            0, 2, 1, 3
        )  # B, S, K, depformer_dim

        dep_outputs = None
        dep_outputs = self.dep_transformer(
            inputs_embeds=inputs_embeds[:, 0, :, :]
        ).last_hidden_state.unsqueeze(1)

        # pass through linears to get logits
        logits = [
            self.linears[i](dep_outputs[:, :, i, :])
            for i in range(cur_num_quantizers)
        ]
        logits = torch.stack(logits, dim=2)  # B, S, K, codebook_size

        return logits

    @property
    def device(self):
        return self.dep_transformer.device


AutoConfig.register(model_type="moshi_bert", config=MoshiTalkingHeadConfig)

from speechlmm.model.adapters.qformer import BertConfig as QformerConfig
from speechlmm.model.adapters.qformer import BertModel as QformerModel


class MoshiNarTalkingHead(TalkingHead):
    _supports_flash_attn_2 = BertModel._supports_flash_attn_2
    _supports_sdpa = BertModel._supports_sdpa

    def __init__(
        self,
        config: MoshiTalkingHeadConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        self.tempformer_dim = self.config.tempformer_dim
        self.depformer_dim = self.config.depformer_dim
        self.depformer_dim_feedforward = self.config.depformer_dim_feedforward
        self.depformer_num_heads = self.config.depformer_num_heads
        self.depformer_num_layers = self.config.depformer_num_layers
        self.depformer_casual = self.config.depformer_casual

        self.norm = "rms_norm"  # FIXME hardcoded
        self.text_vocab_size = self.config.text_vocab_size
        print(
            f"{self.__class__.__name__} initialized with text_vocab_size: {self.text_vocab_size}"
        )

        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=self.norm,
            zero_idx=IGNORE_INDEX,
            dtype=self.torch_dtype,
        )

        # Kwargs
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size

        # Linear layers for depformer_in
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(
                    self.tempformer_dim,
                    self.depformer_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.num_quantizers)
            ]
        )

        # Embeddings for depformer_emb
        self.depformer_emb = nn.ModuleList(
            [
                EmbeddingFactory(self.codebook_size + 1, self.depformer_dim)
                for _ in range(self.num_quantizers - 1)
            ]
        )
        self.depformer_text_emb = EmbeddingFactory(
            self.text_vocab_size, self.depformer_dim
        )

        # Linear layers for output (linears)
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    self.depformer_dim,
                    self.codebook_size,
                    dtype=self.torch_dtype,
                )
                for _ in range(self.num_quantizers)
            ]
        )

        # Bert Transformer initialization
        qformer_config = BertConfig(
            hidden_size=self.depformer_dim,
            num_attention_heads=self.depformer_num_heads,
            intermediate_size=self.depformer_dim_feedforward,
            num_hidden_layers=self.depformer_num_layers,
            is_decoder=self.depformer_casual,  # Adjust for causal architecture
            max_position_embeddings=4096,
            torch_dtype=self.torch_dtype,
        )
        self.qformer = QformerModel(qformer_config).to(dtype=self.torch_dtype)
        # FIXME check if this does not break the model when loading from checkpoint
        self.qformer.cls = None
        self.qformer.embeddings.word_embeddings = None
        self.qformer.embeddings.position_embeddings = None

        # init query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                config.num_query_tokens,
                qformer_config.hidden_size,
                dtype=self.torch_dtype,
            )
        )
        self.query_tokens.data.normal_(
            mean=0.0, std=qformer_config.initializer_range
        )

        self.inner_bs = 1024  # FIXME hardcoded

    def forward(self, context_vector, context_attention_mask, text_tokens):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            context_attention_mask: [B, S]
            text_tokens: [B, S]
            codes: [B, S, K]
        """
        raise NotImplementedError(
            "MoshiNarTalkingHead.forward is not implemented"
        )
        # NOTE: the code below does not handle context_attention_mask
        x = context_vector

        # pass through depformer_in linear layers
        depformer_inputs = [
            self.depformer_in[i](x) for i in range(len(self.depformer_in))
        ]  # B, S, depformer_dim

        text_emb = self.depformer_text_emb(text_tokens)
        depformer_inputs = [
            depformer_inputs[i] + text_emb
            for i in range(len(depformer_inputs))
        ]

        inputs_embeds = torch.stack(
            depformer_inputs, dim=1
        )  # B, K, S, depformer_dim
        inputs_embeds = inputs_embeds.permute(
            0, 2, 1, 3
        )  # B, S, K, depformer_dim

        B, S = context_vector.shape[:2]

        # Flatten the inputs_embeds to B, S*K, depformer_dim

        # Prepare the query tokens along with the attention mask

        # Come on let's twist again! Like we did last summer!
        # chunks = context_vector.size(0) // self.inner_bs + 1
        # query_output_list = []
        # for chunk in range(chunks):
        #     start = self.inner_bs * chunk
        #     end = self.inner_bs * (chunk + 1)
        #     query_embeds = query_tokens[start:end, :, :]
        #     attention_mask_chunk = attention_mask[start:end, :]
        #     if query_embeds.size(0) == 0:
        #         break
        #     query_output = self.qformer(
        #         query_embeds=query_embeds,
        #         attention_mask=attention_mask_chunk,
        #         encoder_hidden_states=???,
        #         encoder_attention_mask=???,
        #         is_decoder=self.depformer_casual,
        #         return_dict=True,
        #     )
        #     last_hidden_state = query_output.last_hidden_state
        #     query_output_list.append(last_hidden_state)

        q_former_output = torch.cat(query_output_list, dim=0)

        # pass through linears to get logits
        logits = [
            self.linears[i](dep_outputs[:, :, i, :])
            for i in range(self.num_quantizers)
        ]
        logits = torch.stack(logits, dim=2)  # B, S, K, codebook_size

        return TalkingHeadOutput(
            logits=logits,
        )

    @torch.no_grad()
    def generate(
        self,
        context_vector,
        text_token,
        sample=False,
        temperature=1.0,
        top_k=0,
        semantic_only=False,
    ):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            text_token: [B, S]
        """
        B, S = context_vector.shape[:2]
        assert B == 1
        assert S == 1

        sequence = text_token.unsqueeze(-1)

        for q in range(self.num_quantizers):
            # print(f"context_vector: {context_vector}")
            # print(f"sequence: {sequence}")
            logits = self.generation_step(context_vector, sequence)[
                :, :, -1, :
            ]
            next_token = sample_token(
                logits.float(),
                use_sampling=False,
                temp=temperature,
                top_k=top_k,
            )
            if semantic_only:
                return torch.cat(
                    [
                        next_token.unsqueeze(-1),
                        torch.full(
                            (1, 1, self.num_quantizers - 1),
                            IGNORE_INDEX,
                            dtype=next_token.dtype,
                            device=next_token.device,
                        ),
                    ],
                    dim=-1,
                )
            sequence = torch.cat([sequence, next_token.unsqueeze(-1)], dim=-1)
        codes = sequence[:, :, 1:]
        return codes

    def generation_step(self, context_vector, sequence):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            sequence: [B, S, K]
        """
        B, S, K = sequence.shape
        assert B == 1
        assert S == 1
        assert K <= self.num_quantizers

        x = context_vector
        cur_num_quantizers = sequence.shape[-1]

        # pass through depformer_in linear layers
        depformer_inputs = [
            self.depformer_in[i](x) for i in range(cur_num_quantizers)
        ]

        # process with depformer_emb embeddings
        last_token_input = []
        for codebook_idx in range(cur_num_quantizers):
            if codebook_idx == 0:
                text_emb = self.depformer_text_emb(sequence[:, :, 0])
                last_token_input.append(text_emb)
            else:
                last_token_input.append(
                    self.depformer_emb[codebook_idx - 1](
                        sequence[:, :, codebook_idx]
                    )
                )

        last_token_input = torch.stack(
            last_token_input, dim=1
        )  # B, K, S, depformer_dim
        depformer_inputs = [
            depformer_inputs[i] + last_token_input[:, i, :]
            for i in range(cur_num_quantizers)
        ]

        inputs_embeds = torch.stack(
            depformer_inputs, dim=1
        )  # B, K, S, depformer_dim
        inputs_embeds = inputs_embeds.permute(
            0, 2, 1, 3
        )  # B, S, K, depformer_dim

        dep_outputs = None
        dep_outputs = self.dep_transformer(
            inputs_embeds=inputs_embeds[:, 0, :, :]
        ).last_hidden_state.unsqueeze(1)

        # pass through linears to get logits
        logits = [
            self.linears[i](dep_outputs[:, :, i, :])
            for i in range(cur_num_quantizers)
        ]
        logits = torch.stack(logits, dim=2)  # B, S, K, codebook_size

        return logits

    @property
    def device(self):
        return self.dep_transformer.device


class NARTalkingHead(TalkingHead):
    _supports_flash_attn_2 = QformerModel._supports_flash_attn_2
    _supports_sdpa = QformerModel._supports_sdpa

    def __init__(
        self,
        config: NARTalkingHeadConfig,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # init qformer and query tokens
        qformer_config = self._init_qformer_and_query_tokens(config)

        # check for text tokens
        if config.depformer_add_text_tokens:
            EmbeddingFactory = partial(
                ScaledEmbedding,
                norm="rms_norm",  # FIXME hardcoded
                zero_idx=IGNORE_INDEX,
                dtype=self.torch_dtype,
            )
            self.depformer_text_emb = EmbeddingFactory(
                self.config.depformer_text_vocab_size,
                qformer_config.hidden_size,
            )

        # config specific
        self.tempformer_dim = self.config.tempformer_dim
        self.depformer_casual = self.config.depformer_casual
        self.num_query_tokens = self.config.num_query_tokens
        self.duration_prediction = self.config.duration_prediction
        self.depformer_multiple_heads = self.config.depformer_multiple_heads
        self.inner_bs = self.config.depformer_inner_batch_size
        self.depformer_expansion_factor = (
            self.config.depformer_expansion_factor
        )
        self.multiple_projection = False  # FIXME hardcoded

        # config super
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size

        # in projection
        if self.multiple_projection:
            self.speech_proj_in = nn.ModuleList(
                [
                    nn.Linear(
                        self.tempformer_dim,
                        qformer_config.hidden_size,
                        dtype=self.torch_dtype,
                    )
                    for _ in range(self.num_quantizers)
                ]
            )
        else:
            self.speech_proj_in = nn.Linear(
                self.tempformer_dim,
                qformer_config.hidden_size,
                dtype=self.torch_dtype,
            )
        # self.speech_proj_in = nn.ModuleList(
        #     [
        #         nn.Linear(
        #             self.tempformer_dim,
        #             qformer_config.hidden_size,
        #             dtype=self.torch_dtype,
        #         )
        #         for _ in range(self.config.cross_attention_window_size)
        #     ]
        # )
        self.ln_speech = nn.LayerNorm(qformer_config.hidden_size)

        # duration prediction
        if self.duration_prediction:
            self.duration_predictor = DurationPredictor(
                qformer_config.hidden_size
            )
            self.duration_prediction_loss_fn = nn.MSELoss()

        # out projection
        if self.depformer_multiple_heads:
            self.codebooks_heads = nn.ModuleList(
                [
                    nn.Linear(
                        qformer_config.hidden_size,
                        self.codebook_size,
                        dtype=self.torch_dtype,
                    )
                    for _ in range(self.num_quantizers)
                ]
            )
        else:
            self.head = nn.Linear(
                qformer_config.hidden_size,
                self.codebook_size,
                dtype=self.torch_dtype,
            )

        self.hop_size = 0
        self.qformer_config = qformer_config

    def generate(
        self,
        context_vector: torch.FloatTensor,
        context_attention_mask: torch.BoolTensor,
        durations_gt: Optional[torch.LongTensor] = None,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        logits = self.forward(
            context_vector, context_attention_mask, durations_gt
        ).logits
        return torch.argmax(logits.softmax(-1), dim=-1)

    def forward(
        self,
        context_vector: torch.FloatTensor,
        context_attention_mask: torch.BoolTensor,
        durations_gt: Optional[torch.LongTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
    ):
        """
        Args:
            context_vector: [B, S, tempformer_dim]
            context_attention_mask: [B, S]
            raw_speech_projector_output: output of the backfeeding speech projector
        """

        # in projection
        if text_tokens is None:
            # pass
            if not self.multiple_projection:
                context_vector = self.ln_speech(
                    self.speech_proj_in(context_vector)
                )
            # context_vector = torch.stack(
            #     [
            #         self.ln_speech(
            #             self.speech_proj_in[i](context_vector)
            #         )
            #         for i in range(self.config.cross_attention_window_size)
            #     ],
            #     dim=2,
            # )
        else:
            # raise NotImplementedError("text_tokens is not None")
            context_vector = self.ln_speech(
                (
                    self.speech_proj_in(context_vector)
                    + self.depformer_text_emb(text_tokens)
                )
            )

        # handle duration prediction
        durations, duration_prediction_loss = self._handle_duration_prediction(
            context_vector, context_attention_mask, durations_gt
        )

        B, T = context_vector.shape[:2]
        if not self.multiple_projection:
            context_vector, context_attention_mask = (
                context_vector.contiguous()
                .view(-1, context_vector.size(-1))
                .unsqueeze(1),
                context_attention_mask.contiguous().view(-1, 1),
            )
        else:
            split_fn_output = self.split_into_windows(
                context_vector, context_attention_mask
            )
            context_vector, context_attention_mask, hop_length = (
                split_fn_output["embeds"],
                split_fn_output["attention_mask"],
                split_fn_output["hop_length"],
            )
        # breakpoint()
        # split_fn_output = self.split_into_windows(
        #     context_vector[:,1:], context_attention_mask[:,1:]
        # )
        # windowed_context_vector, windowed_context_attention_mask, hop_length = (
        #     split_fn_output["embeds"],
        #     split_fn_output["attention_mask"],
        #     split_fn_output["hop_length"],
        # )
        # context_vector = torch.cat(
        #     [
        #         torch.cat(
        #             [
        #                 context_vector[:,0].unsqueeze(1),
        #                 torch.zeros(
        #                     context_vector.shape[0], self.config.cross_attention_window_size-1, context_vector.size(-1)
        #                 ).to(context_vector.device).to(context_vector.dtype)
        #             ], dim=1
        #         ),
        #         windowed_context_vector
        #     ], dim=0
        # )
        # context_attention_mask = torch.cat(
        #     [
        #         torch.cat(
        #             [
        #                 context_attention_mask[:,0].unsqueeze(1),
        #                 torch.zeros(
        #                     context_attention_mask.shape[0], self.config.cross_attention_window_size-1
        #                 ).to(context_attention_mask.device).to(context_attention_mask.dtype)
        #             ], dim=1
        #         ),
        #         windowed_context_attention_mask
        #     ], dim=0
        # )

        # Prepare the query tokens along with the attention mask
        query_tokens = self.query_tokens.expand(context_vector.size(0), -1, -1)
        n_att_repeat = self.num_query_tokens
        attention_mask = (
            (context_attention_mask.sum(dim=1) > 0)
            .view(-1, 1)
            .repeat(1, n_att_repeat)
            .to(torch.long)
            .to(query_tokens.device)
        )
        # Prepare the query tokens with durations if duration prediction is enabled
        if self.duration_prediction and durations is not None:
            query_tokens, attention_mask = (
                self._prepare_query_tokens_with_durations(
                    query_tokens, durations
                )
            )

        # Come on let's twist again! Like we did last summer!
        chunks = context_vector.size(0) // self.inner_bs + 1
        query_output_list = []
        for chunk in range(chunks):
            start = self.inner_bs * chunk
            end = self.inner_bs * (chunk + 1)
            query_embeds = query_tokens[start:end, :, :]
            attention_mask_chunk = attention_mask[start:end, :]
            if query_embeds.size(0) == 0:
                break
            query_output = self.qformer(
                query_embeds=query_embeds,
                attention_mask=attention_mask_chunk,
                encoder_hidden_states=context_vector[start:end, :, :],
                encoder_attention_mask=context_attention_mask[start:end, :],
                is_decoder=self.depformer_casual,
                return_dict=True,
            )
            last_hidden_state = query_output.last_hidden_state
            query_output_list.append(last_hidden_state)

        q_former_output = torch.cat(query_output_list, dim=0)

        # View the query tokens in the same shape as the context vectors
        if self.duration_prediction:
            q_former_output, context_attention_mask = (
                self._pad_query_tokens_with_durations(
                    q_former_output, durations
                )
            )
        else:
            q_former_output, context_attention_mask = (
                self._view_query_tokens_with_compression_factor(
                    B, T, q_former_output, attention_mask
                )
            )

        # project the output to the codebook size
        if self.depformer_multiple_heads:
            q_former_output = torch.stack(
                [
                    codebook(q_former_output[:, :, i, :])
                    for i, codebook in enumerate(self.codebooks_heads)
                ],
                dim=2,
            )
        else:
            q_former_output = self.head(q_former_output)

        # return (q_former_output, context_attention_mask, duration_prediction_loss)
        # FIXME return also duration_prediction_loss
        return TalkingHeadOutput(
            logits=q_former_output,
        )

    def _init_qformer_and_query_tokens(
        self, config: NARTalkingHeadConfig
    ) -> QformerConfig:
        # init qformer
        qformer_config = getattr(config, "qformer", None)
        if qformer_config is None:
            raise ValueError("NARTalkingHead requires a qformer config")
        qformer_config = QformerConfig.from_dict(qformer_config)
        qformer_config.cross_attention_hidden_size = qformer_config.hidden_size
        self.qformer = QformerModel(qformer_config).to(dtype=self.torch_dtype)
        # FIXME check if this does not break the model when loading from checkpoint
        self.qformer.cls = None
        self.qformer.embeddings.word_embeddings = None
        self.qformer.embeddings.position_embeddings = None

        # init query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                config.num_query_tokens,
                qformer_config.hidden_size,
                dtype=self.torch_dtype,
            )
        )
        self.query_tokens.data.normal_(
            mean=0.0, std=qformer_config.initializer_range
        )
        return qformer_config

    def _handle_duration_prediction(
        self,
        context_vector: torch.FloatTensor,
        context_attention_mask: torch.BoolTensor,
        durations_gt: Union[torch.LongTensor, None],
    ):
        duration_prediction_loss = None
        if self.duration_prediction:
            durations_predicted = self.duration_predictor(context_vector)
            if durations_gt is not None:
                duration_prediction_loss = self._compute_duration_loss(
                    durations_gt,
                    durations_predicted,
                    context_vector,
                    context_attention_mask,
                )
            else:
                durations_gt = durations_predicted.round().to(torch.long)
        return durations_gt, duration_prediction_loss

    def _compute_duration_loss(
        self,
        durations_gt: torch.LongTensor,
        durations_predicted: torch.FloatTensor,
        context_vector: torch.FloatTensor,
        context_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        duration_prediction_loss = self.duration_prediction_loss_fn(
            durations_predicted[context_attention_mask].view(-1),
            durations_gt[context_attention_mask]
            .clone()
            .to(context_vector.dtype)
            .view(-1),
        )
        return duration_prediction_loss

    def _prepare_query_tokens_with_durations(self, query_tokens, durations):
        max_duration = durations.long().max().item()
        query_tokens_list, attention_mask_list = [], []
        for query, duration in zip(
            query_tokens,
            durations.contiguous().long().view(-1),
        ):
            query_tokens_list.append(
                torch.cat(
                    [
                        query[: duration.item() * self.num_quantizers, :],
                        torch.zeros(
                            (max_duration - duration.item())
                            * self.num_quantizers,
                            query.size(-1),
                            dtype=query.dtype,
                        ).to(query.device),
                    ]
                )
            )
            attention_mask_list.append(
                torch.cat(
                    [
                        torch.ones(duration.item() * self.num_quantizers),
                        torch.zeros(
                            (max_duration - duration.item())
                            * self.num_quantizers
                        ),
                    ]
                )
            )
        query_tokens = torch.stack(query_tokens_list).to(query_tokens.device)
        attention_mask = (
            torch.stack(attention_mask_list).long().to(query_tokens.device)
        )
        return query_tokens, attention_mask

    def _pad_query_tokens_with_durations(
        self, q_former_output: torch.FloatTensor, durations: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        B = durations.size(0)
        query_output_list, attention_mask_list = [], []
        max_duration = durations.long().sum(-1).max().item()
        q_former_output = q_former_output[attention_mask.bool()]
        for duration in durations.sum(-1):
            query_output_list.append(
                torch.cat(
                    [
                        q_former_output[
                            : duration.item() * self.num_quantizers, :
                        ],
                        torch.zeros(
                            (max_duration - duration.item())
                            * self.num_quantizers,
                            q_former_output.size(-1),
                            dtype=q_former_output.dtype,
                        ).to(q_former_output.device),
                    ],
                    dim=0,
                )
            )
            attention_mask_list.append(
                torch.cat(
                    [
                        torch.ones(duration.item() * self.num_quantizers),
                        torch.zeros(
                            (max_duration - duration.item())
                            * self.num_quantizers
                        ),
                    ],
                    dim=0,
                )
            )
            q_former_output = q_former_output[
                duration.item() * self.num_quantizers :, :
            ]
        assert q_former_output.size(0) == 0

        q_former_output = (
            torch.stack(query_output_list)
            .contiguous()
            .view(
                B,
                max_duration,
                self.num_quantizers,
                q_former_output.size(-1),
            )
        )
        attention_mask = torch.stack(attention_mask_list)
        speech_atts = attention_mask.view(
            B, max_duration, self.num_quantizers
        ).contiguous()
        return q_former_output, speech_atts

    def _view_query_tokens_with_compression_factor(
        self,
        B: int,
        T: int,
        q_former_output: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        T = math.ceil(
            T
            * self.depformer_expansion_factor
            // self.config.cross_attention_window_size
        )  # + #self.depformer_expansion_factor
        q_former_output = q_former_output.view(
            B, T, self.num_quantizers, q_former_output.size(-1)
        )
        attentions = attention_mask.view(B, T, self.num_quantizers)
        return q_former_output, attentions

    def split_into_windows(self, embeds, attention_mask, truncate_last=False):
        batch_size, seq_len, emb_dim = embeds.shape
        window_size_in_frames = self.config.cross_attention_window_size

        excess_frames_in_last_window = seq_len % window_size_in_frames
        if not truncate_last and excess_frames_in_last_window > 0:
            # Zero-pad along the seq_len dimension
            pad_length = window_size_in_frames - excess_frames_in_last_window
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_length))
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_length)
            )

        new_size_in_frames = int(
            round(window_size_in_frames * (1 + self.hop_size))
        )
        hop_length = new_size_in_frames - window_size_in_frames
        stride = (1, window_size_in_frames)
        padding = (0, hop_length)
        kernel_size = (
            1,
            new_size_in_frames,
        )
        unfold_input = embeds.transpose(1, 2).unsqueeze(2)
        windowed_embeds = torch.nn.functional.unfold(
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
        breakpoint()
        windowed_embeds = torch.permute(windowed_embeds, [0, 3, 2, 1])
        embeds = windowed_embeds.reshape(-1, kernel_size[1], emb_dim)
        # ↑ (batch_size * num_windows, window_size_in_frames + hop size, emb_dim)
        windowed_attn = torch.permute(windowed_attn, [0, 2, 1])
        attention_mask = windowed_attn.reshape(-1, new_size_in_frames)

        if truncate_last:
            if self.hop_size:
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


class DurationPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super(DurationPredictor, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(embedding_dim, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc3 = nn.Linear(hidden_dim // 8, 1)  # Single output for duration
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Define the range for the duration
        self.min_duration = 1.0
        self.max_duration = 75.0

    def forward(self, x):
        # Pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # Output layer with sigmoid activation
        x = self.sigmoid(self.fc3(x))
        # Scale the output to the range [min_duration, max_duration]
        duration = self.min_duration + x * (
            self.max_duration - self.min_duration
        )
        return duration
