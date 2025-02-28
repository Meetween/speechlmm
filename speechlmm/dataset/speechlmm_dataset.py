import math
import os
import re
from io import BytesIO
from typing import Optional

import decord
import torch
import torchaudio.transforms as T
import torchvision
from datasets import Dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from speechlmm.arguments import DataArguments
from speechlmm.constants import (
    DEFAULT_AUDIO_OUTPUT_END_TOKEN,
    DEFAULT_AUDIO_OUTPUT_START_TOKEN,
    IGNORE_INDEX,
)
from speechlmm.dataset.custom_dataset.preparers import (
    InterleavedTextAudioNTPPreparer,
)
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.dataset.utils import _tokenize_fn, preprocess
from speechlmm.model.text_audio_aligner import (
    AlignCodesAndText,
    CTCForcedAlignment,
)


class SpeechLmmDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        if dataset is None:
            raise ValueError(
                "Dataset is None. Please provide a valid dataset."
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_args = data_args

        if (
            data_args.group_dataset_by_task["train"]
            and data_args.multi_task_sampler == "length"
        ):
            self.length = dataset["length"]

        if data_args.align_text_to_audio:
            self.aligner = AlignCodesAndText(
                tokenizer=self.tokenizer,
                align_with_whisper=data_args.align_with_whisper,
                restore_punctuation_and_spaces=data_args.restore_punctuation_and_spaces,
            )
        elif data_args.get_timestamps:
            self.ctc_aligner = CTCForcedAlignment()

    def __len__(self):
        return len(self.dataset)

    def process_audio(
        self, audio: torch.Tensor, sr: int, target_sr: int = 16000
    ):
        # assuming audio is mono
        # if len(audio.shape) == 2 and audio.shape[0] > 1:
        #     audio = audio.mean(dim=0, keepdim=True)
        if target_sr is not None:  # handle resampling
            if sr != target_sr:
                audio = T.Resample(sr, target_sr)(audio)
            sr = target_sr
        return audio, sr

    def process_video(
        self, video: dict, target_sr: Optional[int] = None
    ) -> torch.Tensor:
        # decode raw video bytes
        # # TODO(anferico): uncomment once we find a way to use the
        # # GPU-accelerated  version of decord without incurring into that
        # # weird "nvmlInit_v2() failed" error that goes "You should
        # # always run with libnvidia-ml.so that is installed with your
        # # NVIDIA Display Driver [...]"
        # ctx = (
        #     decord.gpu(torch.cuda.current_device())
        #     if torch.cuda.is_available() else decord.cpu(0)
        # )
        ctx = decord.cpu(0)
        video_reader = decord.VideoReader(BytesIO(video["bytes"]), ctx=ctx)

        # read the video frames with a stride that depends on the
        # downsampling factor
        downsampling_factor = 1
        orig_sr = video_reader.get_avg_fps()
        target_sr = target_sr or orig_sr
        if target_sr > orig_sr:
            raise ValueError(
                f"Temporal upsampling is not supported for videos. "
                f"Target SR: {target_sr}, original SR: {orig_sr}"
            )
        downsampling_factor = math.ceil(orig_sr / target_sr)
        video_numpy = video_reader[::downsampling_factor].asnumpy()
        del video_reader

        video = torch.tensor(video_numpy, dtype=torch.uint8)
        video = video.permute(0, 3, 1, 2)  # THWC -> TCHW

        return self._apply_video_transforms(video), target_sr

    def _apply_video_transforms(self, video: torch.Tensor) -> torch.Tensor:
        # TODO(anferico): the hardcoded values below should be
        # configurable
        video = video.to(torch.float32)
        video = torchvision.transforms.functional.normalize(
            video, mean=0, std=255
        )
        video = torchvision.transforms.functional.center_crop(
            video, output_size=88
        )
        video = torchvision.transforms.functional.rgb_to_grayscale(
            video, num_output_channels=1
        )
        video = torchvision.transforms.functional.normalize(
            video, mean=0.421, std=0.165
        )
        return video

    def __getitem__(self, i):
        data = self.dataset[i]
        task = data["task"]
        data_dict = {}

        # Process text data
        if task == "InterleavedTextAudioNTP":
            interleaved_sentence = (
                InterleavedTextAudioNTPPreparer.create_interleaved_text(
                    data["transcription"]
                )
            )
            sources = [
                [
                    {
                        "role": "human",
                        "value": re.sub(
                            DEFAULT_AUDIO_OUTPUT_START_TOKEN
                            + ".*?"
                            + DEFAULT_AUDIO_OUTPUT_END_TOKEN,
                            f"{DEFAULT_AUDIO_OUTPUT_START_TOKEN}{DEFAULT_AUDIO_OUTPUT_END_TOKEN}",
                            interleaved_sentence,
                        ),
                    }
                ]
            ]
        else:
            sources = [data["conversations"]]

        if isinstance(i, int):
            tokenized = preprocess(
                sources, self.tokenizer, self.data_args.conversation_version
            )
            data_dict.update(
                {
                    "input_ids": tokenized["input_ids"][0],
                    "labels": tokenized["labels"][0],
                }
            )

            if "transcription" in data and data["transcription"]:
                transcription_tokens = _tokenize_fn(
                    [data["transcription"]], self.tokenizer
                )
                data_dict["transcription_ids"] = transcription_tokens[
                    "input_ids"
                ][0]
                data_dict["transcription_attention_mask"] = data_dict[
                    "transcription_ids"
                ].new_ones(data_dict["transcription_ids"].shape)

            if "video_transcription" in data and data["video_transcription"]:
                video_transcription_tokens = _tokenize_fn(
                    [data["video_transcription"]], self.tokenizer
                )
                data_dict["video_transcription_ids"] = (
                    video_transcription_tokens["input_ids"][0]
                )
                data_dict["video_transcription_attention_mask"] = data_dict[
                    "video_transcription_ids"
                ].new_ones(data_dict["video_transcription_ids"].shape)

        # Handle audio components
        has_audio_input = (
            "audio_input" in data and data["audio_input"] is not None
        )
        has_audio_output = (
            "audio_output" in data and data["audio_output"] is not None
        )
        has_audio_condition = (
            "audio_condition" in data and data["audio_condition"] is not None
        )

        if has_audio_input:
            audio_input, sr_input = self._process_audio_component(
                data["audio_input"],
                target_sr=self.data_args.audio_input_sampling_rate,
            )
            data_dict.update(
                {"audio_input": [audio_input], "audio_input_sr": [sr_input]}
            )

        audio_output = None
        aligned_tokens = None
        if has_audio_output:
            audio_output, sr_output = self._process_audio_component(
                data["audio_output"],
                target_sr=self.data_args.codec_sampling_rate,
            )
            data_dict.update(
                {
                    "audio_output": [audio_output],
                    "audio_output_sr": [sr_output],
                }
            )

            # Handle alignment
            if task == "InterleavedTextAudioNTP":
                # For interleaved task, align only the audio segment
                audio_output, aligned_tokens = (
                    self._align_interleaved_text_audio(
                        interleaved_sentence=interleaved_sentence,  # Use the generated interleaved text
                        audio_output=audio_output,
                        sr_output=sr_output,
                        output_sentence=data["transcription"],
                        output_language=data["source_language"],
                        alignment_type=(
                            "text_pad_epad"
                            if self.data_args.use_text_tokens
                            else "pad_epad"
                        ),
                    )
                )
            elif self.data_args.align_text_to_audio:
                # Regular alignment for other tasks
                aligned_tokens = self.aligner.align(
                    audio_array=audio_output,
                    sampling_rate=sr_output,
                    frame_rate=self.data_args.codec_frame_rate,
                    sentence=(
                        data["transcription"]
                        if task != "S2ST"
                        else data["text_output"]
                    ),
                    language=data["source_language"],
                    alignment_type=(
                        "text_pad_epad"
                        if self.data_args.use_text_tokens
                        else "pad_epad"
                    ),
                )

            if audio_output is not None:
                data_dict.update({"audio_output": [audio_output]})
            if aligned_tokens is not None:
                data_dict.update(
                    {
                        "aligned_transcription_ids": torch.tensor(
                            aligned_tokens
                        ).to(device=data_dict["input_ids"].device)
                    }
                )

        if has_audio_condition:
            audio_condition, sr_condition = self._process_audio_component(
                data["audio_condition"],
                target_sr=self.data_args.codec_sampling_rate,
                max_duration=self.data_args.max_condition_audio_duration,
            )
            data_dict.update(
                {
                    "audio_condition": [audio_condition],
                    "audio_condition_sr": [sr_condition],
                }
            )

        # Handle video components
        has_video_input = "video_input" in data

        if has_video_input:
            video_input, sr_video_input = self.process_video(
                data["video_input"],
                target_sr=self.data_args.video_input_sampling_rate,
            )
            data_dict.update(
                {
                    "video_input": [video_input],
                    "video_input_sr": [sr_video_input],
                }
            )

        return data_dict

    def _process_audio_component(
        self, audio_data, target_sr, max_duration=None
    ):
        """Helper method to process audio components with optional duration limiting"""
        audio_array = torch.Tensor(audio_data["array"]).to(torch.float32)
        audio, sr = self.process_audio(
            audio_array, audio_data["sampling_rate"], target_sr
        )

        if max_duration and audio.shape[0] > sr * max_duration:
            audio = audio[: sr * max_duration]

        return audio, sr

    def _align_interleaved_text_audio(
        self,
        interleaved_sentence,
        audio_output,
        sr_output,
        output_sentence,
        output_language,
        alignment_type,
    ):
        """Helper method to handle alignment for interleaved text-audio tasks"""
        # Extract text between audio tags
        start_tag = DEFAULT_AUDIO_OUTPUT_START_TOKEN
        end_tag = DEFAULT_AUDIO_OUTPUT_END_TOKEN
        start_pos = interleaved_sentence.find(start_tag) + len(start_tag)
        end_pos = interleaved_sentence.find(end_tag)

        if start_pos == -1 or end_pos == -1:
            raise ValueError("Missing audio tags in interleaved text")

        tagged_segment = interleaved_sentence[start_pos:end_pos].strip()

        # Get word timestamps for the full audio
        word_timestamps = self.aligner.get_word_timestamps(
            audio_array=audio_output,
            sampling_rate=sr_output,
            sentence=output_sentence,
            language=output_language,
        )

        # Extract the audio segment
        audio_segment, segment_timestamps = self._extract_audio_segment(
            tagged_segment,
            audio_output,
            sr_output,
            word_timestamps,
            output_language,
        )

        # Get aligned tokens for the segment
        aligned_tokens = self.aligner.align(
            audio_array=audio_segment,
            sampling_rate=sr_output,
            frame_rate=self.data_args.codec_frame_rate,
            word_timestamps=segment_timestamps,
            language=output_language,
            alignment_type=alignment_type,
        )

        return audio_segment, aligned_tokens

    def _extract_audio_segment(
        self, tagged_text, audio, sr, word_timestamps, output_language
    ):
        """Helper method to extract audio segment corresponding to tagged text"""
        # Process tagged text
        tagged_text_tokens = (
            CTCForcedAlignment.process_text_w_punctuation_and_spaces(
                tagged_text, output_language
            )
        )
        tagged_text_normalized = "".join(tagged_text_tokens)

        # Process full text with same normalization
        text_normalized = "".join([word for _, word in word_timestamps])

        # Try to find exact match
        text_start = text_normalized.find(tagged_text_normalized)

        spaces = True
        if text_start == -1:
            spaces = False
            # Fallback: try without punctuation and spaces
            tagged_text_tokens = CTCForcedAlignment.process_text(
                tagged_text, output_language
            )
            tagged_text_normalized = " ".join(tagged_text_tokens)
            text_normalized = " ".join(
                word.lower() for _, word in word_timestamps
            )
            text_start = text_normalized.find(tagged_text_normalized)
            spaces_count = text_normalized[:text_start].count(" ")
            text_start = text_start - spaces_count
        if text_start == -1:
            raise ValueError(
                "Tagged text segment not found in full transcription"
            )

        text_end = text_start + len(tagged_text_normalized)
        if not spaces:
            text_end = text_end - tagged_text_normalized.count(" ")

        pos = 0
        start_idx = end_idx = None
        for i, (_, word) in enumerate(word_timestamps):
            if start_idx is None and pos + len(word) > text_start:
                start_idx = i
            pos += len(word)
            if pos >= text_end:
                end_idx = i + 1
                break

        if (
            start_idx is None
            or end_idx is None
            or start_idx >= len(word_timestamps)
            or end_idx > len(word_timestamps)
        ):
            raise ValueError(
                f"Invalid word indices: start_idx={start_idx}, end_idx={end_idx}, "
                f"total words={len(word_timestamps)}"
            )

        # Extract aligned tokens for segment
        segment_word_timestamps = word_timestamps[start_idx:end_idx]

        # Calculate audio boundaries
        audio_start = int(segment_word_timestamps[0][0] * sr)
        audio_end = int(
            min(word_timestamps[end_idx][0] * sr, audio.shape[0])
            if end_idx < len(word_timestamps)
            else audio.shape[0]
        )

        # Adjust timestamps relative to segment start
        segment_word_timestamps = [
            (t - segment_word_timestamps[0][0], w)
            for t, w in segment_word_timestamps
        ]

        return audio[audio_start:audio_end], segment_word_timestamps

    def get_data_loader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
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
                batch["video_transcription_ids"] = (
                    batch_video_transcription_ids
                )
            if not all_none(batch_video_transcription_attention_mask):
                batch["video_transcription_attention_mask"] = (
                    batch_video_transcription_attention_mask
                )
            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


class SpeechLmmInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        data_args: DataArguments,
    ):
        if dataset is None:
            raise ValueError(
                "Dataset is None. Please provide a valid dataset."
            )
        self.dataset = dataset
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def process_audio(
        self, audio: torch.Tensor, sr: int, target_sr: int = 16000
    ):
        # assuming audio is mono
        # if len(audio.shape) == 2 and audio.shape[0] > 1:
        #     audio = audio.mean(dim=0, keepdim=True)
        if target_sr is not None:  # handle resampling
            if sr != target_sr:
                audio = T.Resample(sr, target_sr)(audio)
            sr = target_sr
        return audio, sr

    def __getitem__(self, i):
        # process audio and leave the rest as it is
        data = self.dataset[i]

        has_audio_input = data.get("audio_input", None) is not None
        has_audio_output = data.get("audio_output", None) is not None
        has_audio_condition = data.get("audio_condition", None) is not None

        # do not tokenize
        if has_audio_input:
            sr = (
                data["audio_input"]["sampling_rate"]
                if has_audio_input
                else None
            )
            audio_array = torch.Tensor(data["audio_input"]["array"]).to(
                torch.float32
            )
            audio, sr = (
                self.process_audio(
                    audio_array, sr, self.data_args.audio_input_sampling_rate
                )
                if has_audio_input
                else (None, None)
            )
            data["audio_input"] = audio
            data["audio_input_sr"] = sr

        if has_audio_output:
            sr = (
                data["audio_output"]["sampling_rate"]
                if has_audio_output
                else None
            )
            audio_array = torch.Tensor(data["audio_output"]["array"]).to(
                torch.float32
            )
            audio, sr = (
                self.process_audio(
                    audio_array, sr, self.data_args.codec_sampling_rate
                )
                if has_audio_output
                else (None, None)
            )
            data["audio_output"] = audio
            data["audio_output_sr"] = sr

        if has_audio_condition:
            sr = data["audio_condition"]["sampling_rate"]
            audio_array = torch.Tensor(data["audio_condition"]["array"]).to(
                torch.float32
            )
            audio, sr = self.process_audio(
                audio_array, sr, self.data_args.codec_sampling_rate
            )
            # cut audio to 10 seconds if longer - create a new function
            if len(audio.shape) > 1:
                audio = audio.mean(dim=0, keepdim=False)
            MAX_CONDITION_AUDIO_LENGTH = 30
            if audio.shape[0] > sr * MAX_CONDITION_AUDIO_LENGTH:
                audio = audio[: sr * MAX_CONDITION_AUDIO_LENGTH]

            data["audio_condition"] = audio
            data["audio_condition_sr"] = sr

        return data

    def get_data_loader(self, batch_size=1, shuffle=False):
        # return the dict as it is
        def collate_fn(instances):
            if batch_size == 1:
                return instances[0]
            batch = {
                key: [instance[key] for instance in instances]
                for key in instances[0].keys()
            }
            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


import argparse
import os
from pathlib import Path


def optional(type_):
    def parse(value_str):
        if value_str.lower() == "none":
            return None
        return type_(value_str)

    return parse


from transformers import AutoTokenizer

from speechlmm import conversation as conversation_lib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="path_to_config", type=Path, required=True
    )
    parser.add_argument(
        "--audio-input-sampling-rate", type=int, required=False
    )
    parser.add_argument("--codec-sampling-rate", type=int, required=False)
    parser.add_argument("--codec-frame-rate", type=float, required=False)
    parser.add_argument("--num-proc", type=optional(int), default=None)
    parser.add_argument("--rebuild-cache", action="store_true", default=False)
    parser.add_argument(
        "--conv-mode", type=str, required=False, default="llama_3_1"
    )
    args = parser.parse_args()

    data_args = DataArguments(
        audio_input_sampling_rate=args.audio_input_sampling_rate,
        codec_frame_rate=args.codec_frame_rate,
        codec_sampling_rate=args.codec_sampling_rate,
        data_config_path=args.path_to_config,
        is_multimodal=True,
        organize_eval_dataset_per_task=True,
        num_proc_for_preprocessing=args.num_proc,
        rebuild_dataset_cache=args.rebuild_cache,
        align_text_to_audio=True,
        use_text_tokens=True,
        align_with_whisper=False,
        restore_punctuation_and_spaces=True,
    )
    data_args.conversation_version = args.conv_mode
    datasets = DatasetsWrapper(data_args)
    print("Train dataset: ", datasets.train_dataset)

    # model_path = f"{os.getenv('CHECKPOINTS_HOME')}/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_punctuaction/checkpoint-35000"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # dataset = SpeechLmmInferenceDataset(
    #     dataset=datasets.train_dataset,
    #     data_args=data_args,
    # )

    dataset = SpeechLmmDataset(
        dataset=datasets.train_dataset,
        data_args=data_args,
        tokenizer=None,
    )
    dataloader = dataset.get_data_loader(batch_size=1, shuffle=True)
    for i in range(10):
        batch = next(iter(dataloader))
        breakpoint()
