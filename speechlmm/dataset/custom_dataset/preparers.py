import json
import logging
import random
import re
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import librosa
import numpy as np

from speechlmm.constants import (
    DEFAULT_AUDIO_CONDITION_END_TOKEN,
    DEFAULT_AUDIO_CONDITION_START_TOKEN,
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
    DEFAULT_AUDIO_OUTPUT_END_TOKEN,
    DEFAULT_AUDIO_OUTPUT_START_TOKEN,
    DEFAULT_VIDEO_INPUT_END_TOKEN,
    DEFAULT_VIDEO_INPUT_START_TOKEN,
)
from speechlmm.dataset.config import (
    LANGUAGES_CODE_NAME,
    SOURCE_LANGUAGE_PLACEHOLDER,
    TARGET_LANGUAGE_PLACEHOLDER,
    TTS_SENTENCE_PLACEHOLDER,
    SQA_QUESTION_PLACEHOLDER,
    CustomDatasetConfig,
)
from speechlmm.dataset.custom_dataset.base import (
    CaptionMixin,
    ConditioningMixin,
    CustomDataset,
    FewShotsMixin,
    SpokenLanguageUnderstandingDataset,
    get_audio_duration,
)


class TaskPreparer(metaclass=ABCMeta):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "length",
    ]

    def __init__(
        self,
        remap_keys: Optional[Dict[str, str]] = None,
        dataset: Optional[CustomDataset] = None,
        num_few_shots: int = 0,
    ):
        remap_keys = remap_keys or {}
        if hasattr(self, "remap_keys"):
            self.remap_keys.update(remap_keys)
        else:
            self.remap_keys = remap_keys

        self.dataset = dataset
        if num_few_shots > 0 and not isinstance(dataset, FewShotsMixin):
            raise ValueError(
                "`dataset` must extend `FewShotsMixin` if `num_few_shots` > 0"
            )
        self.num_few_shots = num_few_shots

    def prepare_example(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
        is_few_shot_example: bool = False,
    ):
        self._remap_keys_in_example(example)

        self._add_new_fields_in_example(
            example,
            source_language=source_language,
            target_language=target_language,
            config=config,
            is_few_shot_example=is_few_shot_example,
        )

        self._apply_task_specific_transformations(
            example,
            source_language=source_language,
            target_language=target_language,
            config=config,
        )

        if not is_few_shot_example:
            input_, desired_output = self._get_input_and_output(
                example,
                source_language=source_language,
                target_language=target_language,
                config=config,
            )
            example["conversations"] = [
                {
                    "from": "human",
                    "value": input_,
                },
                {"from": "gpt", "value": desired_output},
            ]

        self._add_length_in_example(example)

        for key in list(example.keys()):  # copy keys to avoid RuntimeError
            if key not in self.final_example_fields:
                del example[key]

        return example

    def _remap_keys_in_example(self, example):
        for key, new_key in self.remap_keys.items():
            if key in example:
                example[new_key] = example.pop(key)

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        if (
            "source_language" not in example
            or example["source_language"] is None
        ):
            example["source_language"] = source_language
        if (
            "target_language" not in example
            or example["target_language"] is None
        ):
            example["target_language"] = target_language
        example["task"] = config.task
        if (
            not is_few_shot_example
            and isinstance(self.dataset, FewShotsMixin)
            and self.num_few_shots > 0
        ):
            example["few_shot_examples"] = [
                self.prepare_example(
                    example,
                    source_language=source_language,
                    target_language=target_language,
                    config=config,
                    is_few_shot_example=True,
                )
                for example in self.dataset.get_few_shot_examples(
                    example, n=self.num_few_shots
                )
            ]

    def _add_length_in_example(self, example):
        length = 0
        for conv in example["conversations"]:
            length += len(conv["value"])
        example["length"] = length

    def _apply_task_specific_transformations(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        pass  # no-op by default

    @abstractmethod
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        pass  # no-op by default


class TextInstructPreparer(TaskPreparer):
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        instruction = example["instruction"]
        input_ = example["input"] if "input" in example else ""
        input_ = f"{instruction}\n{input_}" if input_ != "" else instruction
        desired_output = example["output"]

        return input_, desired_output


class NextTokenPredictionPreparer(TaskPreparer):
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        input_ = example["text"]
        return input_, None


class MachineTranslationPreparer(TaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "text_input",
        "text_output",
        "task",
        "few_shot_examples",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        if config.INPUTS_TEXT_LIST is None:
            raise ValueError(
                "Please provide a list of input texts for the model to generate output"
            )

        prompt_language = random.choice(
            list(set([source_language, target_language, "en"]))
        )
        input_, output_prefix = "", ""
        input_ = random.choice(
            config.INPUTS_TEXT_LIST[config.task][prompt_language]
        )
        input_ = input_.replace(
            SOURCE_LANGUAGE_PLACEHOLDER,
            LANGUAGES_CODE_NAME[prompt_language][source_language],
        ).replace(
            TARGET_LANGUAGE_PLACEHOLDER,
            LANGUAGES_CODE_NAME[prompt_language][target_language],
        )

        if config.OUTPUTS_TEXT_LIST is not None:
            output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][source_language]
            )

        input_ = input_ + example["text_input"]
        desired_output = output_prefix + example["text_output"]
        return input_, desired_output


class AudioTaskPreparer(TaskPreparer, metaclass=ABCMeta):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "length",
    ]

    SEC_TO_TOKENS = 10

    def _add_length_in_example(self, example):
        length = 0
        for conv in example["conversations"]:
            length += len(conv["value"])
        length += example["duration"] * self.SEC_TO_TOKENS
        example["length"] = int(length)


class AudioInputTaskPreparer(AudioTaskPreparer, metaclass=ABCMeta):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_input",
        "transcription",
        "length",
    ]

    def __init__(
        self,
        remap_keys: Optional[Dict[str, str]] = None,
        dataset: Optional[CustomDataset] = None,
        num_few_shots: int = 0,
    ):
        remap_keys = remap_keys or {}
        remap_keys.update({"audio": "audio_input"})
        super().__init__(
            remap_keys=remap_keys, dataset=dataset, num_few_shots=num_few_shots
        )

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )
        example["duration"] = get_audio_duration(example, "audio_input")
        return example

    def _apply_task_specific_transformations(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        target_sr = config.additional_args.get("sampling_rate", None)
        if target_sr is not None:
            orig_sr = example["audio_input"]["sampling_rate"]
            example["audio_input"]["array"] = librosa.resample(
                example["audio_input"]["array"],
                orig_sr=orig_sr,
                target_sr=target_sr,
            )
            example["audio_input"]["sampling_rate"] = target_sr


class AudioOutputTaskPreparer(AudioTaskPreparer, metaclass=ABCMeta):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_output",
        "transcription",
        "speaker_id",
        "length",
    ]

    SEC_TO_TOKENS = 20

    def __init__(
        self,
        remap_keys: Optional[Dict[str, str]] = None,
        dataset: Optional[CustomDataset] = None,
        num_few_shots: int = 0,
    ):
        remap_keys = remap_keys or {}
        remap_keys.update({"audio": "audio_output"})
        super().__init__(
            remap_keys=remap_keys, dataset=dataset, num_few_shots=num_few_shots
        )

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )
        example["duration"] = get_audio_duration(example, "audio_output")
        return example

    def _apply_task_specific_transformations(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        target_sr = config.additional_args.get("sampling_rate", None)
        if target_sr is not None:
            orig_sr = example["audio_output"]["sampling_rate"]
            example["audio_output"]["array"] = librosa.resample(
                example["audio_output"]["array"],
                orig_sr=orig_sr,
                target_sr=target_sr,
            )
            example["audio_output"]["sampling_rate"] = target_sr


class SpeechRecognitionPreparer(AudioInputTaskPreparer):
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = random.choice([source_language, "en"])
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        return (
            DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + input_,
            desired_output_prefix + example["transcription"],
        )


class SpeechTranslationPreparer(AudioInputTaskPreparer):
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = random.choice(
            [source_language, target_language, "en"]
        )
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
            input_ = input_.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source_language],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target_language],
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )
            desired_output_prefix = desired_output_prefix.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source_language],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target_language],
            )

        return (
            DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + input_,
            desired_output_prefix + example["translation"],
        )


class VideoInputTaskPreparer(TaskPreparer, metaclass=ABCMeta):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "video_input",
        "video_transcription",
    ]

    def __init__(
        self,
        remap_keys: Optional[Dict[str, str]] = None,
        dataset: Optional[CustomDataset] = None,
        num_few_shots: int = 0,
    ):
        remap_keys = remap_keys or {}
        remap_keys.update(
            {
                "video": "video_input",
                "transcription": "video_transcription",
            }
        )
        super().__init__(
            remap_keys=remap_keys, dataset=dataset, num_few_shots=num_few_shots
        )


class VisualSpeechRecognitionPreparer(VideoInputTaskPreparer):
    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = random.choice([source_language, "en"])
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        return (
            DEFAULT_VIDEO_INPUT_START_TOKEN
            + DEFAULT_VIDEO_INPUT_END_TOKEN
            + "\n"
            + input_,
            desired_output_prefix + example["video_transcription"],
        )


class TextToSpeechPreparer(AudioOutputTaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_output",
        "audio_condition",
        "transcription",
        "speaker_id",
        "length",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        omit_conditioning = False
        add_utterance_description = config.additional_args.get(
            "add_speaker_descriptions", False
        )
        utterance_description_subprompt = ""
        if (
            isinstance(self.dataset, CaptionMixin)
            and add_utterance_description
        ):
            utterance_description = self.dataset.get_caption_for_example(
                example
            ).strip()
            utterance_description_subprompt = (
                "\n" + f"Speech description: {utterance_description}"
            )
            condition_on_speaker_prob = config.additional_args.get(
                "condition_on_speaker_prob", 1.0
            )
            omit_conditioning = random.random() >= condition_on_speaker_prob

        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            conditioning_template_string = ""
            if (
                isinstance(self.dataset, ConditioningMixin)
                and not omit_conditioning
            ):
                conditioning_template_string = (
                    "\n"
                    + DEFAULT_AUDIO_CONDITION_START_TOKEN
                    + DEFAULT_AUDIO_CONDITION_END_TOKEN
                )

            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][
                    "en"
                ]  # FIXME: add support for other languages
            )
            input_ = (
                input_.replace(
                    TTS_SENTENCE_PLACEHOLDER, example["transcription"]
                )
                + utterance_description_subprompt
                + conditioning_template_string
            )

        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = (
                random.choice(
                    config.OUTPUTS_TEXT_LIST[config.task][source_language]
                )
                + "\n"
            )

        return (
            input_,
            desired_output_prefix
            + DEFAULT_AUDIO_OUTPUT_START_TOKEN
            + DEFAULT_AUDIO_OUTPUT_END_TOKEN,
        )

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )

        if isinstance(self.dataset, ConditioningMixin):
            conditioning_example = self.dataset.get_conditioning_for_example(
                example
            )
            example["audio_condition"] = conditioning_example["audio"]
            example["duration"] = example["duration"] + min(
                get_audio_duration(example, "audio_condition"), 10
            )


class TextToSpeechCaptionOnlyPreparer(AudioOutputTaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_output",
        "transcription",
        "speaker_id",
        "length",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        add_utterance_description = config.additional_args.get(
            "add_speaker_descriptions", False
        )
        utterance_description_subprompt = ""
        if (
            isinstance(self.dataset, CaptionMixin)
            and add_utterance_description
        ):
            utterance_description = self.dataset.get_caption_for_example(
                example
            ).strip()
            utterance_description_subprompt = (
                "\n" + f"Speech description: {utterance_description}"
            )

        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][source_language]
            )
            input_ = (
                input_.replace(
                    TTS_SENTENCE_PLACEHOLDER, example["transcription"]
                )
                + utterance_description_subprompt
            )

        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = (
                random.choice(
                    config.OUTPUTS_TEXT_LIST[config.task][source_language]
                )
                + "\n"
            )

        return (
            input_,
            desired_output_prefix
            + DEFAULT_AUDIO_OUTPUT_START_TOKEN
            + DEFAULT_AUDIO_OUTPUT_END_TOKEN,
        )


class SpeechToSpeechTranslationPreparer(TaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_input",
        "audio_output",
        "text_input",
        "text_output",
        "speaker_id",
        "audio_condition",
        "length",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        if target_language is None:
            target_language = "en"

        prompt_language = random.choice(
            [source_language, target_language, "en"]
        )

        if prompt_language is None:
            prompt_language = "en"
            logging.warning(f"prompt_language is None, set to en - {example}")

        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
            input_ = input_.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source_language],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target_language],
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )
            desired_output_prefix = desired_output_prefix.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source_language],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target_language],
            )

        add_translation_prefix = config.additional_args.get(
            "add_translation_prefix", False
        )
        if add_translation_prefix:
            desired_output_prefix += example["text_output"]

        audio_input_ = (
            DEFAULT_AUDIO_INPUT_START_TOKEN + DEFAULT_AUDIO_INPUT_END_TOKEN
        )
        audio_output_ = (
            DEFAULT_AUDIO_OUTPUT_START_TOKEN + DEFAULT_AUDIO_OUTPUT_END_TOKEN
        )
        audio_condition_ = ""

        if (
            "audio_condition" in example
            and example["audio_condition"] is not None
        ):
            audio_condition_ = (
                DEFAULT_AUDIO_CONDITION_START_TOKEN
                + DEFAULT_AUDIO_CONDITION_END_TOKEN
                + "\n"
            )

        input_, output_ = (
            audio_input_ + "\n" + audio_condition_ + input_,
            desired_output_prefix + audio_output_,
        )

        return input_, output_

    # TODO(anferico): this isn't really a "task-specific transformation"
    def _apply_task_specific_transformations(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        example["audio_input"]["array"] = np.asarray(
            example["audio_input"]["array"]
        )
        example["audio_output"]["array"] = np.asarray(
            example["audio_output"]["array"]
        )

        # resample both audio input and output
        target_sr = config.additional_args.get("sampling_rate", None)
        if target_sr is not None:
            orig_sr_audio_input = example["audio_input"]["sampling_rate"]
            example["audio_input"]["array"] = librosa.resample(
                example["audio_input"]["array"],
                orig_sr=orig_sr_audio_input,
                target_sr=target_sr,
            )
            example["audio_input"]["sampling_rate"] = target_sr

            orig_sr_audio_output = example["audio_output"]["sampling_rate"]
            example["audio_output"]["array"] = librosa.resample(
                example["audio_output"]["array"],
                orig_sr=orig_sr_audio_output,
                target_sr=target_sr,
            )
            example["audio_output"]["sampling_rate"] = target_sr

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )
        durations = [
            get_audio_duration(example, "audio_input"),
            get_audio_duration(example, "audio_output"),
        ]

        if isinstance(self.dataset, ConditioningMixin):
            conditioning_example = self.dataset.get_conditioning_for_example(
                example
            )
            example["audio_condition"] = conditioning_example["audio"]
            durations.append(
                min(get_audio_duration(example, "audio_condition"), 10)
            )

        example["duration"] = sum(durations)


# TODO(anferico): shouldn't this be a subclass of AudioOutputTaskPreparer?
class TextToSpeechBasePreparer(TaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_output",
        "transcription",
        "speaker_id",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        return (
            DEFAULT_AUDIO_OUTPUT_START_TOKEN + DEFAULT_AUDIO_OUTPUT_END_TOKEN,
            None,
        )


class InterleavedTextAudioNTPPreparer(TaskPreparer):
    """
    Preparer for next token prediction with interleaved text and audio tokens.
    """

    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "audio_output",
        "audio_condition",
        "transcription",
        "speaker_id",
        "interleaved_text",
    ]

    @staticmethod
    def create_interleaved_text(transcription: str) -> tuple[str, int, int]:
        """
        Creates interleaved text with audio tags by randomly selecting a segment of text.

        Args:
            transcription: The full transcription text

        Returns:
            tuple containing:
            - interleaved_text: Text with audio tags inserted
            - start_idx: Starting word index of audio segment
            - num_words: Number of words in audio segment
        """
        words = transcription.split()
        assert len(words) >= 20
        # Select a random segment to convert to audio
        min_words = 5
        max_words = 15
        num_words = (
            random.randint(min_words, max_words)
            if len(words) > min_words
            else len(words)
        )
        start_idx = random.randint(0, len(words) - num_words)

        # Create interleaved text with audio tags
        prefix = " ".join(words[:start_idx])
        audio_segment = " ".join(words[start_idx : start_idx + num_words])
        suffix = " ".join(words[start_idx + num_words :])

        interleaved_text = (
            (prefix + " " if prefix else "")
            + DEFAULT_AUDIO_OUTPUT_START_TOKEN
            + audio_segment
            + DEFAULT_AUDIO_OUTPUT_END_TOKEN
            + (" " + suffix if suffix else "")
        ).strip()

        return interleaved_text

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        """Creates input with interleaved text and audio tokens."""
        interleaved_text = example["interleaved_text"]
        # remove text beetween <audio_output_start> and <audio_output_end>
        interleaved_text = re.sub(
            DEFAULT_AUDIO_OUTPUT_START_TOKEN
            + ".*?"
            + DEFAULT_AUDIO_OUTPUT_END_TOKEN,
            f"{DEFAULT_AUDIO_OUTPUT_START_TOKEN}{DEFAULT_AUDIO_OUTPUT_END_TOKEN}",
            interleaved_text,
        )
        return (
            interleaved_text,
            "",
        )  # Empty output since this is for next token prediction

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )

        example["interleaved_text"] = self.create_interleaved_text(
            example["transcription"]
        )


# TODO(anferico):
# - add <few_shot_audio_start> and <few_shot_audio_end> tokens to be
#   replaced by the actual audio features in the DataLoader
class SpokenLanguageUnderstandingPreparer(AudioInputTaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_input",
        "transcription",
        "length",
        "intent",
        "transcription_with_annotated_slots",
    ]

    def __init__(
        self,
        remap_keys: Optional[Dict[str, str]] = None,
        dataset: Optional[SpokenLanguageUnderstandingDataset] = None,
        do_slot_filling: bool = True,
        num_few_shots: int = 0,
    ):
        if not isinstance(dataset, SpokenLanguageUnderstandingDataset):
            # NOTE: `None` is not a valid value either
            raise ValueError(
                "`dataset` must be a `SpokenLanguageUnderstandingDataset`"
            )

        super().__init__(
            remap_keys=remap_keys, dataset=dataset, num_few_shots=num_few_shots
        )
        self.do_slot_filling = do_slot_filling

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )

        if self.do_slot_filling:
            example["transcription_with_annotated_slots"] = (
                self.dataset.get_transcription_with_annotated_slots(
                    transcription=example["transcription"], example=example
                )
            )

    def get_formatted_output(self, example, version):
        if version == 1:
            output = {"intent_str": example["intent"]}
            if self.do_slot_filling:
                output["annot_utt"] = example[
                    "transcription_with_annotated_slots"
                ]
            return json.dumps(output)
        elif version == 2:
            output = f"intent_str: {example['intent']}"
            if self.do_slot_filling:
                for slot in example["slots"]:
                    slot_type, slot_value = slot.split("=")
                    output += f", {slot_type}: {slot_value}"
            return output
        elif version == 3:
            if self.do_slot_filling:
                return f"{example['intent']}\n{example['transcription_with_annotated_slots']}"
            else:
                return example["intent"]
        else:
            raise ValueError("Invalid version")

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        if config.task == "SLU":
            assert (
                self.do_slot_filling
            ), "`do_slot_filling` must be True for the SLU task"
        elif config.task == "SLU_INTENT_ONLY":
            assert (
                not self.do_slot_filling
            ), "`do_slot_filling` must be False for the SLU_INTENT_ONLY task"
        else:
            raise ValueError(f"Invalid task: {config.task}")

        # TODO(anferico): at the moment, we keep the prompt in English
        # and explicitly tell the model the language of the input speech.
        # This is because the intent names and slot types we have in
        # SLURP and SpeechMassive are in English
        prompt_language = "en"
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            slu_task_specification = config.INPUTS_TEXT_LIST[config.task][
                prompt_language
            ]
            output_format_version = slu_task_specification.get(
                "output_format_version", 3
            )

            prompt = slu_task_specification["task_explanation"] + "\n\n"

            intents = self.dataset.get_intents()
            if len(intents) > 0:
                prompt += "Intent strings:\n"
                prompt += (
                    "\n".join(f"- {intent}" for intent in intents) + "\n\n"
                )

            if self.do_slot_filling:
                slot_types = self.dataset.get_slot_types()
                if len(slot_types) > 0:
                    prompt += "Slot types:\n"
                    prompt += (
                        "\n".join(f"- {slot}" for slot in slot_types) + "\n\n"
                    )

            if "input_format" in slu_task_specification:
                prompt += (
                    "Input format:\n"
                    + slu_task_specification["input_format"]
                    + "\n\n"
                )

            if "output_format" in slu_task_specification:
                prompt += (
                    "Output format:\n"
                    + slu_task_specification["output_format"]
                    + "\n\n"
                )

            if "example_output" in slu_task_specification:
                prompt += (
                    f"For instance, if the utterance says "
                    f"\"{slu_task_specification['example_input_transcription']}\", "
                    f"then the output should be:\n"
                    + slu_task_specification["example_output"]
                    + "\n\n"
                )

            if "few_shot_examples" in example:
                prompt += (
                    "What follows are some examples of input and output pairs. "
                    "Note that, in these examples, you are NOT provided with "
                    "the actual spoken utterances, but only with their textual "
                    "transcriptions. In real scenarios, you will be provided "
                    "with spoken utterances but NOT their transcriptions.\n"
                )
                for i, few_shot_example in enumerate(
                    example["few_shot_examples"], start=1
                ):
                    input = (
                        "Spoken utterance: "
                        + few_shot_example["transcription"]
                        + "\n"
                        + "Language of the spoken utterance: "
                        + LANGUAGES_CODE_NAME["en"][
                            few_shot_example["source_language"]
                        ]
                    )

                    output = self.get_formatted_output(
                        few_shot_example, output_format_version
                    )

                    prompt += f"Example input {i}:\n{input}\n\n"
                    prompt += f"Example output {i}:\n{output}\n\n"

            if "instructions" in slu_task_specification:
                prompt += "Instructions:\n"
                prompt += (
                    "\n".join(
                        f"{i}. {instruction}"
                        for i, instruction in enumerate(
                            slu_task_specification["instructions"], start=1
                        )
                    )
                    + "\n\n"
                )

            if source_language not in LANGUAGES_CODE_NAME["en"]:
                raise ValueError(
                    f"Please add {source_language} to LANGUAGES_CODE_NAME['en']"
                )

            language = LANGUAGES_CODE_NAME["en"][source_language]
            input_ = (
                prompt
                + "Spoken utterance: "
                + DEFAULT_AUDIO_INPUT_START_TOKEN
                + DEFAULT_AUDIO_INPUT_END_TOKEN
                + "\n"
                + f"Language of the spoken utterance: {language}\n"
            )

        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )
            desired_output_prefix = desired_output_prefix.replace(
                SOURCE_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][source_language],
            ).replace(
                TARGET_LANGUAGE_PLACEHOLDER,
                LANGUAGES_CODE_NAME[prompt_language][target_language],
            )

        desired_output = self.get_formatted_output(
            example, output_format_version
        )
        return input_, desired_output_prefix + desired_output


class MultiTurnTextInstructPreparer(TaskPreparer):
    """Preparer for multi-turn instruction conversations."""

    def prepare_example(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
        add_conversations: bool = True,
    ):
        """Prepare a multi-turn conversation example."""
        # Add standard fields
        example["source_language"] = source_language
        example["target_language"] = target_language
        example["task"] = config.task

        if add_conversations:
            # Convert messages array into conversation format
            example["conversations"] = []
            for message in example["messages"]:
                example["conversations"].append(
                    {"from": message["role"], "value": message["content"]}
                )

        # Clean up intermediate fields
        for key in ["messages", "message_tree_id", "num_messages"]:
            example.pop(key, None)

        self._add_length_in_example(example)

        return example

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        """Not used since we handle conversation formatting in prepare_example."""
        raise NotImplementedError("Use prepare_example instead")


class SpokenQuestionAnsweringPreparer(AudioInputTaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_input",
        "transcription",
        "question",
        "answer",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = source_language
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
            input_ = input_.replace(
                SQA_QUESTION_PLACEHOLDER, example["question"]
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        return (
            DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + input_,
            desired_output_prefix + example["answer"],
        )


class SpeechSummarizationPreparer(AudioInputTaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "audio_input",
        "transcription",
        "summary",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = source_language
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        return (
            DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + input_,
            desired_output_prefix + example["summary"],
        )


class TextSummarizationPreparer(TaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "text_input",
        "text_output",
        "summary",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = source_language
        input_, desired_output_prefix = "", ""
        if config.INPUTS_TEXT_LIST is not None:
            input_ = random.choice(
                config.INPUTS_TEXT_LIST[config.task][prompt_language]
            )
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        return (
            input_,
            desired_output_prefix + example["summary"],
        )


class SpeechToSpeechInstructionFollowingPreparer(TaskPreparer):
    final_example_fields = [
        "conversations",
        "source_language",
        "target_language",
        "task",
        "few_shot_examples",
        "audio_input",
        "audio_output",
        "text_input",
        "text_output",
        "speaker_id",
        "audio_condition",
    ]

    def _get_input_and_output(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        prompt_language = source_language

        desired_output_prefix = ""
        if config.OUTPUTS_TEXT_LIST is not None:
            desired_output_prefix = random.choice(
                config.OUTPUTS_TEXT_LIST[config.task][prompt_language]
            )

        audio_input_ = (
            DEFAULT_AUDIO_INPUT_START_TOKEN + DEFAULT_AUDIO_INPUT_END_TOKEN
        )
        audio_output_ = (
            DEFAULT_AUDIO_OUTPUT_START_TOKEN + DEFAULT_AUDIO_OUTPUT_END_TOKEN
        )
        audio_condition_ = ""

        if (
            "audio_condition" in example
            and example["audio_condition"] is not None
        ):
            audio_condition_ = (
                DEFAULT_AUDIO_CONDITION_START_TOKEN
                + DEFAULT_AUDIO_CONDITION_END_TOKEN
            )

        input_, output_ = (
            audio_input_ + "\n" + audio_condition_,
            desired_output_prefix + audio_output_,
        )

        return input_, output_

    # TODO(anferico): this isn't really a "task-specific transformation"
    def _apply_task_specific_transformations(
        self,
        example,
        source_language: str,
        target_language: str,
        config: CustomDatasetConfig,
    ):
        example["audio_input"]["array"] = np.asarray(
            example["audio_input"]["array"]
        )
        example["audio_output"]["array"] = np.asarray(
            example["audio_output"]["array"]
        )

        # resample both audio input and output
        target_sr = config.additional_args.get("sampling_rate", None)
        if target_sr is not None:
            orig_sr_audio_input = example["audio_input"]["sampling_rate"]
            example["audio_input"]["array"] = librosa.resample(
                example["audio_input"]["array"],
                orig_sr=orig_sr_audio_input,
                target_sr=target_sr,
            )
            example["audio_input"]["sampling_rate"] = target_sr

            orig_sr_audio_output = example["audio_output"]["sampling_rate"]
            example["audio_output"]["array"] = librosa.resample(
                example["audio_output"]["array"],
                orig_sr=orig_sr_audio_output,
                target_sr=target_sr,
            )
            example["audio_output"]["sampling_rate"] = target_sr

    def _add_new_fields_in_example(
        self,
        example,
        source_language,
        target_language,
        config,
        is_few_shot_example=False,
    ):
        super()._add_new_fields_in_example(
            example,
            source_language,
            target_language,
            config,
            is_few_shot_example=is_few_shot_example,
        )

        if isinstance(self.dataset, ConditioningMixin):
            conditioning_example = self.dataset.get_conditioning_for_example(
                example
            )
            example["audio_condition"] = conditioning_example["audio"]
