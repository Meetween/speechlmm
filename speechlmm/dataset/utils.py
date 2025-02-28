import copy
import hashlib
import itertools
import math
import multiprocessing
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from unittest import mock

import datasets
import filelock
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset, load_dataset
from datasets.utils import tqdm as hf_tqdm
from datasets.utils.py_utils import convert_file_size_to_int
from torch.utils.data import BatchSampler, Sampler

from speechlmm import conversation as conversation_lib
from speechlmm.arguments import DataArguments
from speechlmm.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
)
from speechlmm.mm_utils import tokenizer_mm_token
from speechlmm.model import *

IS_TOKENIZER_GREATER_THAN_0_14 = transformers.__version__ >= "4.0.0"

import logging

from joblib import Parallel, delayed


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str], data_args: DataArguments
) -> Dict:
    """This function preprocesses multimodal data by moving multimodal tokens to the beginning of the sentence."""
    logging.warning("preprocess_multimodal is deprecated")
    if not data_args.is_multimodal:
        return sources

    # TODO: avoid moving multimodal tokens to the beginning of the sentence,
    # instead use start/end tags (adding special tokens to llm vocab) and
    # leave multimodal tokens in their original position
    multimodal_tokens = [
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_AUDIO_TOKEN,
        DEFAULT_VIDEO_TOKEN,
    ]  # TODO: this should be defined in constants.py
    for source in sources:
        for sentence in source:
            for mm_token in multimodal_tokens:
                # modality = "image" if mm_token == DEFAULT_IMAGE_TOKEN else "audio"
                if mm_token in sentence["value"]:
                    num_tokens = sentence["value"].count(mm_token)
                    sentence["value"] = mm_token + sentence["value"].replace(
                        mm_token, ""
                    )
                    replace_tokens = f"{mm_token}" * num_tokens + "\n"
                    sentence["value"] = replace_tokens + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    if (
                        "mmtag"
                        in conversation_lib.default_conversation.version
                    ):
                        raise NotImplementedError("mmtag is not supported")
                    # if data_args.mm_use_im_start_end: # TODO must be replaced by mm_use_start_end_tags
                    # raise NotImplementedError("mm_use_im_start_end is not supported")
                    # if "image" in modality: start_tag = DEFAULT_IM_START_TOKEN; end_tag = DEFAULT_IM_END_TOKEN
                    # elif "audio" in modality: start_tag = DEFAULT_AUDIO_START_TOKEN; end_tag = DEFAULT_AUDIO_END_TOKEN
                    # sentence["value"] = sentence["value"].replace(
                    #     mm_token, start_tag + mm_token + end_tag
                    # )
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if (
                i != 0
                and not tokenizer.legacy
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1
            instruction_len = len(tokenizer(parts[0]).input_ids)

            if i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3_base(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocesses text for basic next-token prediction with Llama 3.

    Unlike other preprocessing functions that handle conversational data,
    this function is designed for simple next-token prediction on raw text.
    No conversation templates or system prompts are applied.

    Args:
        sources: List of tuples, each containing a user message and a system message - here we only use the user message of the first tuple
        tokenizer: The tokenizer to use for converting text to token IDs

    Returns:
        Dict containing:
            - input_ids: Padded tensor of token IDs
            - labels: Clone of input_ids with padding tokens masked as IGNORE_INDEX

    Note:
        This is specifically for basic language modeling tasks where we want
        the model to predict each next token given the previous tokens,
        rather than following a conversational format.
    """

    is_output_present = (
        len(sources[0]) > 1
        and sources[0][1]["value"] is not None
        and sources[0][1]["value"] != ""
    )

    input_ = sources[0][0]["value"]
    output = sources[0][1]["value"] if is_output_present else None
    prompt = input_ + output if is_output_present else input_

    # logging.info(f"input: {input_}")
    # logging.info(f"output: {output}")
    # logging.info(f"prompt: {prompt}")

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    # mask pad tokens and source[0]["value"]
    targets[targets == tokenizer.pad_token_id] = IGNORE_INDEX
    if is_output_present:
        # mask the tokens corresponding to the input part of the prompt
        targets[:, : len(tokenizer(input_).input_ids)] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # FIXME: this function is not used anymore
    raise NotImplementedError("This function is not used anymore")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert (
            DEFAULT_IMAGE_TOKEN in source[0]["value"]
            or DEFAULT_AUDIO_TOKEN in source[0]["value"]
            or DEFAULT_VIDEO_TOKEN in source[0]["value"]
        )
        default_token = None
        if DEFAULT_IMAGE_TOKEN in source[0]["value"]:
            default_token = DEFAULT_IMAGE_TOKEN
        elif DEFAULT_AUDIO_TOKEN in source[0]["value"]:
            default_token = DEFAULT_AUDIO_TOKEN
        elif DEFAULT_VIDEO_TOKEN in source[0]["value"]:
            default_token = DEFAULT_VIDEO_TOKEN
        else:
            raise NotImplementedError(
                f"found bad token in {source[0]['value']}"
            )

        source[0]["value"] = default_token
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)

    # tokenize conversations
    input_ids = [
        tokenizer_mm_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_mm_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    conversation_version: str,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    conversation = conversation_lib.conv_templates[conversation_version]
    conversation_lib.default_conversation = conversation
    if conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(
            sources,
            tokenizer,
        )
    if conversation.version.startswith("v1"):
        return preprocess_v1(
            sources,
            tokenizer,
        )
    if conversation.version == "mpt":
        return preprocess_mpt(
            sources,
            tokenizer,
        )
    if conversation.version == "llama_3_1":
        return preprocess_llama3(
            sources,
            tokenizer,
        )
    if conversation.version == "llama_3_1_base":
        return preprocess_llama3_base(
            sources,
            tokenizer,
        )
    conversations = []
    for source in sources:
        header = f"{conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["value"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # FIXME: THIS IMPLEMENTATION DOESN'T ALLOW FOR MIXED DATA
        # AS FOR THE MULTIMODAL DATA IT ONLY CHECKS THE TYPE
        # IN THE FIRST ELEMENT OF THE BATCH

        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # handling etherogeneous samples in the batch, if a key is not present in the batch, add None in the index corresponding to the sample
        batch_tasks = [None] * len(instances)
        batch_input_audios_srs = [None] * len(instances)
        batch_output_audios_srs = [None] * len(instances)
        batch_condition_audios_srs = [None] * len(instances)
        batch_transcription_ids = [None] * len(instances)
        batch_transcription_attention_mask = [None] * len(instances)
        batch_aligned_transcription_ids = [None] * len(instances)
        batch_input_videos_srs = [None] * len(instances)
        batch_video_transcription_ids = [None] * len(instances)
        batch_video_transcription_attention_mask = [None] * len(instances)
        for i, instance in enumerate(instances):
            if "audio_input" in instance:
                batch_input_audios_srs[i] = (
                    instance["audio_input"][0],
                    instance["audio_input_sr"][0],
                )
            if "audio_output" in instance:
                batch_output_audios_srs[i] = (
                    instance["audio_output"][0],
                    instance["audio_output_sr"][0],
                )
            if "audio_condition" in instance:
                batch_condition_audios_srs[i] = (
                    instance["audio_condition"][0],
                    instance["audio_condition_sr"][0],
                )
            if "transcription_ids" in instance:
                batch_transcription_ids[i] = instance["transcription_ids"]
            if "transcription_attention_mask" in instance:
                batch_transcription_attention_mask[i] = instance[
                    "transcription_attention_mask"
                ]
            if "aligned_transcription_ids" in instance:
                batch_aligned_transcription_ids[i] = instance[
                    "aligned_transcription_ids"
                ]
            if "video_input" in instance:
                batch_input_videos_srs[i] = (
                    instance["video_input"][0],
                    instance["video_input_sr"][0],
                )
            if "video_transcription_ids" in instance:
                batch_video_transcription_ids[i] = instance[
                    "video_transcription_ids"
                ]
            if "video_transcription_attention_mask" in instance:
                batch_video_transcription_attention_mask[i] = instance[
                    "video_transcription_attention_mask"
                ]

        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_input_audios_srs):
            batch["input_audios_srs"] = batch_input_audios_srs
        if not all_none(batch_output_audios_srs):
            batch["output_audios_srs"] = batch_output_audios_srs
        if not all_none(batch_condition_audios_srs):
            batch["condition_audios_srs"] = batch_condition_audios_srs
        if not all_none(batch_transcription_ids):
            batch["transcription_ids"] = batch_transcription_ids
        if not all_none(batch_transcription_attention_mask):
            batch["transcription_attention_mask"] = (
                batch_transcription_attention_mask
            )
        if not all_none(batch_aligned_transcription_ids):
            batch["aligned_transcription_ids"] = (
                batch_aligned_transcription_ids
            )
        if not all_none(batch_input_videos_srs):
            batch["input_videos_srs"] = batch_input_videos_srs
        if not all_none(batch_video_transcription_ids):
            batch["video_transcription_ids"] = batch_video_transcription_ids
        if not all_none(batch_video_transcription_attention_mask):
            batch["video_transcription_attention_mask"] = (
                batch_video_transcription_attention_mask
            )

        return batch


def get_audio_modality_lengths(dataset):
    logging.warning(
        "get_audio_modality_lengths is bypassed by a list of zeros"
    )
    return [0] * len(dataset)
    audio_modality_len = []
    for example in dataset:
        duration = 0
        if "audio_input" in example and example["audio_input"] is not None:
            duration += (
                len(example["audio_input"]["array"])
                / example["audio_input"]["sampling_rate"]
            )
        elif "audio_output" in example and example["audio_output"] is not None:
            duration += (
                len(example["audio_output"]["array"])
                / example["audio_output"]["sampling_rate"]
            )
        elif (
            "audio_condition" in example
            and example["audio_condition"] is not None
        ):
            duration += (
                len(example["audio_condition"]["array"])
                / example["audio_condition"]["sampling_rate"]
            )
        audio_modality_len.append(duration)
    return audio_modality_len


NormalizableDataType = Union[dict, list, str, int, float, bool, None]


def _normalize_data(value: NormalizableDataType) -> NormalizableDataType:
    if isinstance(value, dict):
        # sort dictionary items and recursively normalize values
        return tuple((k, _normalize_data(v)) for k, v in sorted(value.items()))
    elif isinstance(value, (list, tuple)):
        # sort items after normalizing each element
        return tuple(sorted(_normalize_data(x) for x in value))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise ValueError(f"Unsupported type: {type(value)}")


def get_dict_fingerprint(dict_: dict) -> str:
    normalized_dict = _normalize_data(dict_)
    dict_bytes = str(normalized_dict).encode("utf-8")
    return hashlib.md5(dict_bytes).hexdigest()


def save_parquet_shards(
    dataset: datasets.Dataset,
    dataset_dir: Union[str, Path],
    max_shard_size: Optional[Union[str, int]] = None,
    num_shards: Optional[int] = None,
    num_proc: Optional[int] = None,
):
    if max_shard_size is not None and num_shards is not None:
        raise ValueError(
            "Please specify either max_shard_size or num_shards, but not both."
        )
    if dataset.list_indexes():
        raise ValueError(
            "please remove all the indexes using `dataset.drop_index` before saving a dataset"
        )

    num_proc = min(num_proc or 1, len(dataset))
    if num_shards is None:
        dataset_nbytes = dataset._estimate_nbytes()
        max_shard_size = convert_file_size_to_int(max_shard_size or "500MB")
        num_shards = int(dataset_nbytes / max_shard_size) + 1
        num_shards = max(num_shards, num_proc)

    if num_proc == 1:
        for shard_index in hf_tqdm(
            range(num_shards),
            unit="shards",
            desc="Saving Parquet shards",
        ):
            _save_single_parquet_shard(
                dataset, dataset_dir, shard_index, num_shards
            )
    else:
        with multiprocessing.Pool(num_proc) as pool:
            pool.starmap(
                _save_single_parquet_shard,
                hf_tqdm(
                    [
                        (dataset, dataset_dir, shard_index, num_shards)
                        for shard_index in range(num_shards)
                    ],
                    unit="shards",
                    desc=f"Saving Parquet shards (num_proc={num_proc})",
                ),
            )


def _save_single_parquet_shard(dataset, dataset_dir, shard_index, num_shards):
    shard = dataset.shard(
        index=shard_index, num_shards=num_shards, contiguous=True
    )
    shard_path = Path(
        dataset_dir, f"{shard_index:05d}_of_{num_shards-1:05d}.parquet"
    )

    def no_tqdm(iterable, *args, **kwargs):
        return iterable

    with mock.patch("datasets.utils.tqdm", new=no_tqdm):
        shard.to_parquet(str(shard_path))


def load_parquet_shards(dataset_dir, num_proc=None):
    """Load parquet shards with improved error handling and timeout."""
    # Hardcode a timeout of 60 seconds in `BaseFileLock.acquire` to
    # prevent indefinite waiting
    acquire_original = filelock._api.BaseFileLock.acquire

    def acquire_patched(
        self_BaseFileLock,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        poll_intervall: float | None = None,
        blocking: bool | None = None,
    ):
        return acquire_original(
            self_BaseFileLock,
            timeout=60,  # 60 second timeout
            poll_interval=poll_interval,
            poll_intervall=poll_intervall,
            blocking=blocking,
        )

    with mock.patch("filelock._api.BaseFileLock.acquire", acquire_patched):
        dataset_dict = load_dataset(str(dataset_dir), num_proc=num_proc)
        return dataset_dict["train"]  # default split name


class RandomMultiTaskSampler(Sampler[int]):
    """Samples elements randomly from multiple tasks based on specified weights.
    Ensures that samples are grouped to guarantee each batch will have at least
    one sample from each task.

    Args:
        data_source (Dict[str, Dataset]): Dictionary mapping task names to their datasets
        task_weights (Dict[str, float]): Dictionary mapping task names to their sampling weights (0-1)
        batch_size (int): Size of mini-batch to ensure proper grouping
        replacement (bool): If True, samples with replacement to match weights exactly
        num_samples (Optional[int]): Total number of samples when replacement=False
        seed (int, optional): Random seed for reproducibility
    """

    TASK_CATEGORIES = {
        "text": ["TextInstruct", "MultiTurnTextInstruct", "MT"],
        "audio_input": ["ASR", "ST"],
        "audio_output": ["TTS", "S2ST"],
    }

    def __init__(
        self,
        data_source: Dict[str, Dataset],
        task_weights: Dict[str, float],
        batch_size: int,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        epoch: int = 0,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self.epoch = epoch
        self.batch_size = batch_size

        # Validate and normalize weights
        total_weight = sum(task_weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError("Task weights must sum to 1.0")

        if not replacement and num_samples is None:
            raise ValueError(
                "num_samples must be specified when replacement=False"
            )

        self.task_weights = task_weights
        self.task_indices = {}
        self.task_sizes = {}
        self.length = 0

        # Create index mappings and store sizes
        for task_name, dataset in data_source.items():
            task_size = int(len(dataset) * task_weights[task_name])
            self.task_indices[task_name] = list(
                range(self.length, self.length + task_size)
            )
            self.task_sizes[task_name] = task_size
            self.length += task_size

        # Adjust num_samples to ensure it's divisible by batch_size
        if not self.replacement:
            self.num_samples = (num_samples // batch_size) * batch_size
        else:
            self.num_samples = (self.length // batch_size) * batch_size

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed + epoch)

    def _create_balanced_batch_indices(self, num_batches: int) -> List[int]:
        """Creates indices ensuring each batch has at least one sample from each task."""
        all_indices = []
        tasks = list(self.task_weights.keys())

        for _ in range(num_batches):
            batch_indices = []

            # Ensure that each category is sampled at least once
            for category in self.TASK_CATEGORIES.keys():
                tasks_per_category = set(self.TASK_CATEGORIES[category])
                training_tasks = set(self.task_weights.keys())
                # check if there is at least one task in the category that is present in the task_weights
                if not tasks_per_category.intersection(training_tasks):
                    continue
                # random sample one task from the category that is present in the task_weights and then sample from the task
                rnd_task = random.choice(
                    list(tasks_per_category.intersection(training_tasks))
                )

                task_idx = self.task_indices[rnd_task]
                if self.replacement:
                    batch_indices.append(
                        task_idx[
                            torch.randint(
                                len(task_idx), (1,), generator=self.generator
                            ).item()
                        ]
                    )
                else:
                    available_indices = list(set(task_idx) - set(all_indices))
                    if not available_indices:
                        available_indices = task_idx
                    batch_indices.append(
                        available_indices[
                            torch.randint(
                                len(available_indices),
                                (1,),
                                generator=self.generator,
                            ).item()
                        ]
                    )

            # Fill remaining batch slots according to weights
            remaining_slots = self.batch_size - len(
                self.TASK_CATEGORIES.keys()
            )  # - len(tasks)
            if remaining_slots > 0:
                weights = torch.tensor(
                    [self.task_weights[task] for task in tasks]
                )
                remaining_samples = torch.multinomial(
                    weights,
                    remaining_slots,
                    replacement=True,
                    generator=self.generator,
                )

                for task_idx in remaining_samples:
                    task = tasks[task_idx]
                    task_indices = self.task_indices[task]
                    if self.replacement:
                        idx = task_indices[
                            torch.randint(
                                len(task_indices),
                                (1,),
                                generator=self.generator,
                            ).item()
                        ]
                    else:
                        available_indices = list(
                            set(task_indices) - set(all_indices)
                        )
                        if not available_indices:
                            available_indices = task_indices
                        idx = available_indices[
                            torch.randint(
                                len(available_indices),
                                (1,),
                                generator=self.generator,
                            ).item()
                        ]
                    batch_indices.append(idx)

            # Shuffle the batch indices
            batch_indices = torch.tensor(batch_indices)[
                torch.randperm(len(batch_indices), generator=self.generator)
            ]
            all_indices.extend(batch_indices.tolist())

        return all_indices

    def __iter__(self):
        num_batches = self.num_samples // self.batch_size
        indices = self._create_balanced_batch_indices(num_batches)

        self._log_distribution(indices)
        return iter(indices)

    def _log_distribution(self, indices):
        logging.info(f"Logging distribution of {len(indices)} indices")

        # Create index to task mapping once
        index_to_task = {}
        for task, task_idx in self.task_indices.items():
            for idx in task_idx:
                index_to_task[idx] = task

        # Count using direct lookup
        task_counts = {task: 0 for task in self.task_weights}
        for idx in indices:
            task = index_to_task[idx]
            task_counts[task] += 1

        total_samples = sum(task_counts.values())
        for task, count in task_counts.items():
            percentage = (count / total_samples) * 100
            target_percentage = self.task_weights[task] * 100
            logging.info(
                f"{task}: {count} samples ({percentage:.1f}% vs target {target_percentage:.1f}%)"
            )

    def __len__(self):
        return self.num_samples if not self.replacement else self.length


class LengthGroupedMultiTaskSampler(RandomMultiTaskSampler):

    def __init__(
        self,
        data_source: Dict[str, Dataset],
        task_weights: Dict[str, float],
        batch_size: int,
        world_size: int,
        variable_batch_size: bool = False,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        epoch: int = 0,
    ):
        # Fix parameter order in super().__init__
        super().__init__(
            data_source, task_weights, replacement, num_samples, seed, epoch
        )
        self.batch_size = batch_size if not variable_batch_size else None
        self.world_size = world_size
        self.variable_batch_size = variable_batch_size

        # Get lengths for each dataset
        self.lengths = []
        for task_name, dataset in data_source.items():
            task_lengths = dataset.length
            self.lengths.extend(task_lengths)

    def __iter__(self):
        if self.replacement:
            # Calculate target samples for each task based on weights
            target_samples = {
                task: int(self.task_weights[task] * self.length)
                for task in self.task_weights
            }

            # Generate indices for each task with replacement
            all_indices = []
            for task, target in target_samples.items():
                task_idx = self.task_indices[task]
                task_size = len(task_idx)

                # Generate random indices with replacement
                rand_indices = torch.randint(
                    0, task_size, (target,), generator=self.generator
                )
                all_indices.extend([task_idx[i] for i in rand_indices])

            # Shuffle all indices
            all_indices = torch.tensor(all_indices)[
                torch.randperm(len(all_indices), generator=self.generator)
            ].tolist()
        else:
            all_indices = list(self._iter_without_replacement())

        # Group by length and distribute across GPUs
        if not self.variable_batch_size:
            indices = self._get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                indices=all_indices,
                generator=self.generator,
            )
        else:
            indices = self._get_sorted_indices(
                self.lengths, indices=all_indices, generator=self.generator
            )

        self._log_distribution(indices)
        return iter(indices)

    def _get_sorted_indices(self, lengths, indices=None, generator=None):
        """Groups indices by length for more efficient processing."""
        if indices is None:
            indices = list(range(len(lengths)))

        # Sort indices by length
        sorted_indices = sorted(
            indices, key=lambda i: lengths[i], reverse=True
        )

        return sorted_indices

    def _get_length_grouped_indices(
        self, lengths, batch_size, world_size, indices=None, generator=None
    ):
        """Groups indices by length for more efficient processing."""
        if indices is None:
            indices = list(range(len(lengths)))

        # Calculate mega-batch size (batch_size * world_size)
        megabatch_size = world_size * batch_size

        # Split indices into mega-batches
        megabatches = [
            indices[i : i + megabatch_size]
            for i in range(0, len(indices), megabatch_size)
        ]

        # Sort each mega-batch by length
        megabatches = [
            sorted(megabatch, key=lambda i: lengths[i], reverse=True)
            for megabatch in megabatches
        ]

        # Split each sorted mega-batch into world_size chunks
        megabatches = [
            self._split_to_even_chunks(megabatch, lengths, world_size)
            for megabatch in megabatches
        ]

        # Flatten the chunks into a single list
        return [
            i
            for megabatch in megabatches
            for batch in megabatch
            for i in batch
        ]

    def _split_to_even_chunks(self, indices, lengths, num_chunks):
        """Split indices into chunks of similar total length."""
        if len(indices) == 0:
            return [[] for _ in range(num_chunks)]

        # Handle case where indices can't be evenly divided
        if len(indices) < num_chunks:
            return [indices[i : i + 1] for i in range(len(indices))] + [
                [] for _ in range(num_chunks - len(indices))
            ]

        if len(indices) % num_chunks != 0:
            return [indices[i::num_chunks] for i in range(num_chunks)]

        # Calculate target number of indices per chunk
        num_indices_per_chunk = len(indices) // num_chunks

        # Initialize chunks and their running lengths
        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]

        # Distribute indices to minimize length variance between chunks
        for index in indices:
            # Find chunk with shortest total length
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]

            # Mark chunk as full when it reaches target size
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")

        return chunks


class SequentialMultiTaskSampler(Sampler[int]):
    """Samples elements sequentially from multiple tasks while respecting weights.

    Key features:
    1. Maintains sequential access within each task
    2. Handles small datasets with large weights through controlled repetition
    3. Handles large datasets with small weights through proportional sampling
    4. Interleaves tasks based on their weights

    Args:
        data_source (Dict[str, Dataset]): Dictionary mapping task names to their datasets
        task_weights (Dict[str, float]): Dictionary mapping task names to their sampling weights (0-1)
        seed (int, optional): Random seed for reproducibility
    """

    def __init__(
        self,
        data_source: Dict[str, Dataset],
        task_weights: Dict[str, float],
        seed: Optional[int] = None,
    ):
        self.data_source = data_source

        # Validate and normalize weights
        total_weight = sum(task_weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError("Task weights must sum to 1.0")

        self.task_weights = task_weights
        self.task_indices = {}
        self.task_sizes = {}
        self.length = 0

        # Create index mappings and store sizes
        for task_name, dataset in data_source.items():
            task_size = len(dataset)
            self.task_indices[task_name] = list(
                range(self.length, self.length + task_size)
            )
            self.task_sizes[task_name] = task_size
            self.length += task_size

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # Calculate target samples for each task based on weights
        target_samples = {
            task: int(self.task_weights[task] * self.length)
            for task in self.task_weights
        }

        # Calculate how many times to repeat each dataset
        repetition_factors = {}
        task_positions = {}  # Track current position in each task
        for task, target in target_samples.items():
            dataset_size = self.task_sizes[task]
            if target > dataset_size:
                # Small dataset, large weight: need repetition
                repetition_factors[task] = math.ceil(target / dataset_size)
            else:
                # Large dataset, small weight: no repetition needed
                repetition_factors[task] = 1
            task_positions[task] = 0

        # Calculate interleaving pattern
        all_indices = []
        remaining_samples = {
            task: target_samples[task] for task in self.task_weights
        }

        # Create proportional chunks based on weights
        while sum(remaining_samples.values()) > 0:
            for task in self.task_weights:
                if remaining_samples[task] > 0:
                    # Calculate chunk size proportional to weight
                    chunk_size = max(1, int(self.task_weights[task] * 10))
                    chunk_size = min(chunk_size, remaining_samples[task])

                    # Get indices for this chunk
                    task_idx = self.task_indices[task]
                    for _ in range(chunk_size):
                        if task_positions[task] >= len(task_idx):
                            # Reset position if we need to repeat
                            if repetition_factors[task] > 1:
                                task_positions[task] = 0
                            else:
                                break

                        all_indices.append(task_idx[task_positions[task]])
                        task_positions[task] += 1
                        remaining_samples[task] -= 1

                        if remaining_samples[task] == 0:
                            break

        # Log distribution
        task_counts = {task: 0 for task in self.task_weights}
        for idx in all_indices:
            for task, indices in self.task_indices.items():
                if idx in indices:
                    task_counts[task] += 1
                    break

        total_samples = sum(task_counts.values())
        logging.info("Sequential sampling distribution:")
        for task, count in task_counts.items():
            percentage = (count / total_samples) * 100
            target_percentage = self.task_weights[task] * 100
            logging.info(
                f"{task}: {count} samples ({percentage:.1f}% vs target {target_percentage:.1f}%)"
            )

        return iter(all_indices)

    def __len__(self):
        return self.length


class MultiTaskBatchSampler(BatchSampler):
    """Creates batches from indices provided by a MultiTask sampler.

    Args:
        sampler (Sampler): Base sampler (Sequential or Random MultiTask sampler)
        batch_size (int): Size of mini-batch
        drop_last (bool): If True, drop the last incomplete batch
    """

    def __init__(
        self, sampler: Sampler, batch_size: int, drop_last: bool = False
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class MultiSourceDistributedBatchSampler(Sampler[List[int]]):
    """
    Custom batch sampler that supports multiple datasets, even for different
    tasks. Returns batches from each dataset in a round-robin fashion,
    occasionally breaking this pattern in case the datasets don't have the same
    number of samples.
    """

    def __init__(
        self,
        per_device_batch_size: int,
        dataset_lengths: Sequence[int],
        dataset_weights: Optional[Sequence[float]] = None,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if dataset_weights is not None and len(dataset_lengths) != len(
            dataset_weights
        ):
            raise ValueError(
                "`dataset_lengths` and `dataset_weights` must have the same length."
            )

        self.dataset_lengths = dataset_lengths
        self.dataset_weights = dataset_weights
        self.per_device_batch_size = per_device_batch_size
        self.drop_last = drop_last

        self.shuffle = shuffle

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch = 0

        self.batch_size = self.per_device_batch_size * num_replicas
        self.num_batches = None
        self.num_datasets = len(self.dataset_lengths)

        self.weighted_dataset_lengths = (
            self._compute_weighted_dataset_lengths()
            if self.dataset_weights is not None
            else self.dataset_lengths
        )
        self.num_batches_per_dataset = [
            self._compute_number_of_batches_for_dataset(length)
            for length in self.weighted_dataset_lengths
        ]

    def _compute_weighted_dataset_lengths(self):
        # TODO(anferico): consider other strategies for computing the weighted dataset lengths
        total_weighted_length = sum(
            length * weight
            for length, weight in zip(
                self.dataset_lengths, self.dataset_weights
            )
        )
        return [
            math.ceil(total_weighted_length * weight)
            for weight in self.dataset_weights
        ]

    def __len__(self):
        if self.num_batches is None:
            self.num_batches = sum(self.num_batches_per_dataset)
        return self.num_batches

    def _compute_number_of_batches_for_dataset(self, dataset_length: int):
        rounding_fn = math.floor if self.drop_last else math.ceil
        return rounding_fn(dataset_length / self.batch_size)

    def __iter__(self):
        sample_indices_per_dataset = []
        cumulative_length = 0
        for dataset_index in range(self.num_datasets):
            original_length = self.dataset_lengths[dataset_index]
            final_length = self.weighted_dataset_lengths[dataset_index]
            quotient, remainder = divmod(final_length, original_length)
            sample_indices_for_this_dataset = []
            while quotient > 0:
                if self.shuffle:
                    indices = (
                        torch.randperm(
                            original_length, generator=self.generator
                        )
                        + cumulative_length
                    )
                    indices = indices.tolist()
                else:
                    indices = list(
                        range(
                            cumulative_length,
                            cumulative_length + original_length,
                        )
                    )
                sample_indices_for_this_dataset.extend(indices)
                quotient -= 1
            if remainder > 0:
                if self.shuffle:
                    indices = (
                        torch.randperm(
                            original_length, generator=self.generator
                        )[:remainder]
                        + cumulative_length
                    )
                    indices = indices.tolist()
                else:
                    indices = list(
                        range(cumulative_length, cumulative_length + remainder)
                    )
                sample_indices_for_this_dataset.extend(indices)
            sample_indices_per_dataset.append(sample_indices_for_this_dataset)
            cumulative_length += original_length

        dataset_batch_sequence = self._make_alternating_batches()

        current_batch_start_idx_per_dataset = [0] * self.num_datasets
        for dataset_index in dataset_batch_sequence:
            indices = sample_indices_per_dataset[dataset_index]
            batch_idx = current_batch_start_idx_per_dataset[dataset_index]
            batch = indices[batch_idx : batch_idx + self.batch_size]
            if len(batch) > 0 and (
                not self.drop_last or len(batch) == self.batch_size
            ):
                yield batch

            current_batch_start_idx_per_dataset[dataset_index] += len(batch)

    def _make_alternating_batches(self):
        num_complete_alternating_batches = min(self.num_batches_per_dataset)
        complete_batches = [
            torch.randperm(
                self.num_datasets, generator=self.generator
            ).tolist()
            for _ in range(num_complete_alternating_batches)
        ]

        remaining_indices = []
        for dataset_index, num_batches in enumerate(
            self.num_batches_per_dataset
        ):
            indices_to_add = num_batches - num_complete_alternating_batches
            remaining_indices.extend([dataset_index] * indices_to_add)

        permutation = torch.randperm(
            len(remaining_indices), generator=self.generator
        )
        remaining_indices = [
            remaining_indices[i] for i in permutation.tolist()
        ]

        def batched(iterable, n):
            if n < 1:
                raise ValueError("`n` must be ≥ 1")
            iterator = iter(iterable)
            batches = []
            while batch := list(itertools.islice(iterator, n)):
                batches.append(batch)
            return batches

        incomplete_batches = batched(remaining_indices, self.num_datasets)
        partially_filled_last_batch = None
        if (
            len(incomplete_batches) > 0
            and len(incomplete_batches[-1]) < self.num_datasets
        ):
            partially_filled_last_batch = incomplete_batches.pop()

        all_batches = complete_batches + incomplete_batches
        permutation = torch.randperm(
            len(all_batches), generator=self.generator
        )
        all_batches = [all_batches[i] for i in permutation.tolist()]

        if partially_filled_last_batch is not None:
            all_batches.append(partially_filled_last_batch)

        dataset_batch_sequence = []
        for batch in all_batches:
            dataset_batch_sequence.extend(batch)

        return dataset_batch_sequence

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


# NOTE(anferico): to be used with split_batches=False (DOES NOT WORK)
# class _MultiSourceDistributedBatchSampler(Sampler[List[int]]):
#     """
#     Custom batch sampler that supports multiple datasets, even for different
#     tasks. Returns batches from each dataset in a round-robin fashion,
#     occasionally breaking this pattern in case the datasets don't have the same
#     number of samples.
#     """
#     def __init__(
#         self,
#         batch_size: int,
#         dataset_lengths: Sequence[int],
#         dataset_weights: Optional[Sequence[float]] = None,
#         drop_last_batch_if_not_full: bool = False,
#         num_replicas: Optional[int] = None,
#         rank: Optional[int] = None,
#         shuffle: bool = True,
#         seed: int = 0,
#         drop_samples_if_not_evenly_distributable: bool = False,
#     ):
#         if (
#             dataset_weights is not None
#             and len(dataset_lengths) != len(dataset_weights)
#         ):
#             raise ValueError(
#                 "`dataset_lengths` and `dataset_weights` must have the same length."
#             )

#         self.dataset_lengths = dataset_lengths
#         self.dataset_weights = dataset_weights
#         self.batch_size = batch_size
#         self.drop_last_batch_if_not_full = drop_last_batch_if_not_full
#         self.drop_samples_if_not_evenly_distributable = (
#             drop_samples_if_not_evenly_distributable
#         )

#         self.shuffle = shuffle

#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError(
#                     "Requires distributed package to be available"
#                 )
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError(
#                     "Requires distributed package to be available"
#                 )
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
#             )

#         self.num_replicas = num_replicas
#         self.rank = rank

#         self.generator = torch.Generator()
#         self.generator.manual_seed(seed)
#         self.epoch = 0

#         self.total_batch_size = self.batch_size * self.num_replicas
#         self.num_batches = None
#         self.num_datasets = len(self.dataset_lengths)

#         weighted_dataset_lengths = (
#             self._compute_weighted_dataset_lengths()
#             if self.dataset_weights is not None
#             else self.dataset_lengths
#         )
#         self.final_dataset_lengths = [
#             self._make_length_even_across_replicas(length)
#             for length in weighted_dataset_lengths
#         ]
#         self.num_batches_per_dataset = [
#             self._compute_number_of_batches_for_dataset(length)
#             for length in self.final_dataset_lengths
#         ]

#     def _compute_weighted_dataset_lengths(self):
#         # TODO(anferico): consider other strategies for computing the weighted dataset lengths
#         total_weighted_length = sum(
#             length * weight for length, weight in zip(
#                 self.dataset_lengths, self.dataset_weights
#             )
#         )
#         return [
#             math.ceil(total_weighted_length * weight)
#             for weight in self.dataset_weights
#         ]

#     def _make_length_even_across_replicas(self, dataset_length: int):
#         num_samples_in_even_splits, remainder = divmod(
#             dataset_length, self.num_replicas
#         )
#         if remainder > 0 and not self.drop_samples_if_not_evenly_distributable:
#             return dataset_length + (self.num_replicas - remainder)
#         return num_samples_in_even_splits * self.num_replicas

#     def __len__(self):
#         if self.num_batches is None:
#             self.num_batches = sum(self.num_batches_per_dataset)
#         return self.num_batches

#     def _compute_number_of_batches_for_dataset(
#         self, dataset_length: int
#     ):
#         rounding_fn = (
#             math.floor if self.drop_last_batch_if_not_full else math.ceil
#         )
#         return rounding_fn(dataset_length / self.total_batch_size)

#     def __iter__(self):
#         sample_indices_per_dataset = []
#         cumulative_length = 0
#         for dataset_index in range(self.num_datasets):
#             original_length = self.dataset_lengths[dataset_index]
#             final_length = self.final_dataset_lengths[dataset_index]
#             quotient, remainder = divmod(final_length, original_length)
#             sample_indices_for_this_dataset = []
#             while quotient > 0:
#                 if self.shuffle:
#                     indices = torch.randperm(original_length, generator=self.generator) + cumulative_length
#                     indices = indices.tolist()
#                 else:
#                     indices = list(range(cumulative_length, cumulative_length + original_length))
#                 sample_indices_for_this_dataset.extend(indices)
#                 quotient -= 1
#             if remainder > 0:
#                 if self.shuffle:
#                     # indices = self.rng.sample(
#                     #     range(
#                     #         cumulative_length,
#                     #         cumulative_length + original_length,
#                     #     ),
#                     #     remainder
#                     # )
#                     indices = torch.randperm(original_length, generator=self.generator)[:remainder] + cumulative_length
#                     indices = indices.tolist()
#                 else:
#                     indices = list(
#                         range(cumulative_length, cumulative_length + remainder)
#                     )
#                 sample_indices_for_this_dataset.extend(indices)
#             sample_indices_per_dataset.append(sample_indices_for_this_dataset)
#             cumulative_length += original_length

#         dataset_batch_sequence = self._make_alternating_batches()

#         current_batch_start_idx_per_dataset = [0] * self.num_datasets
#         for dataset_index in dataset_batch_sequence:
#             slice_size = self.final_dataset_lengths[dataset_index] // self.num_replicas
#             start_idx = self.rank * slice_size
#             end_idx = start_idx + slice_size
#             indices = sample_indices_per_dataset[dataset_index][start_idx:end_idx]

#             batch_idx = current_batch_start_idx_per_dataset[dataset_index]
#             batch = indices[batch_idx:batch_idx+self.batch_size]
#             if (
#                 len(batch) > 0
#                 and (
#                     not self.drop_last_batch_if_not_full
#                     or len(batch) == self.batch_size
#                 )
#             ):
#                 yield batch

#             current_batch_start_idx_per_dataset[dataset_index] += len(batch)

#     def _make_alternating_batches(self):
#         num_complete_alternating_batches = min(self.num_batches_per_dataset)
#         complete_batches = [
#             # self.rng.sample(range(self.num_datasets), k=self.num_datasets)
#             torch.randperm(self.num_datasets, generator=self.generator).tolist()
#             for _ in range(num_complete_alternating_batches)
#         ]

#         remaining_indices = []
#         for dataset_index, num_batches in enumerate(
#             self.num_batches_per_dataset
#         ):
#             indices_to_add = num_batches - num_complete_alternating_batches
#             remaining_indices.extend([dataset_index] * indices_to_add)

#         permutation = torch.randperm(len(remaining_indices), generator=self.generator)
#         remaining_indices = [remaining_indices[i] for i in permutation.tolist()]
#         # self.rng.shuffle(remaining_indices)

#         def batched(iterable, n):
#             if n < 1:
#                 raise ValueError("`n` must be ≥ 1")
#             iterator = iter(iterable)
#             batches = []
#             while batch := list(itertools.islice(iterator, n)):
#                 batches.append(batch)
#             return batches

#         incomplete_batches = batched(remaining_indices, self.num_datasets)
#         partially_filled_last_batch = None
#         if (
#             len(incomplete_batches) > 0
#             and len(incomplete_batches[-1]) < self.num_datasets
#         ):
#             partially_filled_last_batch = incomplete_batches.pop()

#         all_batches = complete_batches + incomplete_batches
#         # self.rng.shuffle(all_batches)
#         permutation = torch.randperm(len(all_batches), generator=self.generator)
#         all_batches = [all_batches[i] for i in permutation.tolist()]
#         if partially_filled_last_batch is not None:
#             all_batches.append(partially_filled_last_batch)

#         dataset_batch_sequence = []
#         for batch in all_batches:
#             dataset_batch_sequence.extend(batch)

#         return dataset_batch_sequence

#     def set_epoch(self, epoch: int) -> None:
#         r"""
#         Set the epoch for this sampler.

#         When :attr:`shuffle=True`, this ensures all replicas
#         use a different random ordering for each epoch. Otherwise, the next iteration of this
#         sampler will yield the same ordering.

#         Args:
#             epoch (int): Epoch number.
#         """
#         self.epoch = epoch

if __name__ == "__main__":
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    # Create dummy datasets
    class DummyDataset(Dataset):
        def __init__(self, size, task_name):
            self.size = size
            self.task_name = task_name

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"{self.task_name}_{idx}"

    # Create three dummy datasets of different sizes
    datasets = {
        "small": DummyDataset(1, "small"),  # Small dataset
        "medium": DummyDataset(10, "medium"),  # Medium dataset
        "large": DummyDataset(1000000, "large"),  # Large dataset
    }

    task_weights = {
        "small": 0.1,  # Large weight despite small size
        "medium": 0.6,  # Balanced weight
        "large": 0.3,  # Small weight despite large size
    }

    def test_sampler(sampler_class, **kwargs):
        print(f"\nTesting {sampler_class.__name__}")
        sampler = sampler_class(datasets, task_weights, **kwargs)
        batch_sampler = MultiTaskBatchSampler(
            sampler, batch_size=8, drop_last=False
        )
        dataloader = DataLoader(
            ConcatDataset(list(datasets.values())), batch_sampler=batch_sampler
        )

        # Count samples per task
        task_counts = {"small": 0, "medium": 0, "large": 0}
        items_processed = []
        flag = True
        for i, batch in enumerate(dataloader):
            batch_task_counts = {"small": 0, "medium": 0, "large": 0}
            # Count task occurrences
            for item in batch:
                items_processed.append(item)
                task_name = item.split("_")[0]
                task_counts[task_name] += 1
                batch_task_counts[task_name] += 1

            # print batch distribution
            print(f"Batch {i}: {batch_task_counts}")
            for task, count in batch_task_counts.items():
                if count == 0:
                    flag = False
                    break
        if flag:
            print("All batches have samples from all tasks")
        else:
            print("Some batches do not have samples from all tasks")
        print(f"items_processed: {len(items_processed)}")

        # count the number of repeated samples
        repeated_samples = len(items_processed) - len(set(items_processed))

        print(f"Number of repeated samples: {repeated_samples}")
        # Print statistics
        total_samples = sum(task_counts.values())
        print(f"\nTask distribution in first {i} batches:")
        for task, count in task_counts.items():
            percentage = (count / total_samples) * 100
            print(f"{task}: {count} samples ({percentage:.1f}%)")

    # Test both samplers
    # test_sampler(RandomMultiTaskSampler, replacement=False, seed=42, num_samples=300000)
    # print("replacement=False")
    test_sampler(
        RandomMultiTaskSampler, replacement=True, seed=42, batch_size=8
    )
    print("replacement=True")
    # test_sampler(SequentialMultiTaskSampler)
