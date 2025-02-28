import logging
import random
from collections import defaultdict

import numpy as np
from datasets import DownloadMode, concatenate_datasets, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from speechlmm.constants import (
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
)
from speechlmm.dataset.config import (
    LANGUAGES_CODE_NAME,
    SLU_INTENT_STR,
    SLU_N_FEW_SHOT,
    SLU_SLOT_TYPE,
    CustomDatasetConfig,
)


class Slurp(Dataset):
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()

        self.intent_str = SLU_INTENT_STR[self.__class__.__name__]
        self.slot_type = SLU_SLOT_TYPE[self.__class__.__name__]

        # remove intents not present in SLU_INTENT_STR["SpeechMassive"]
        self.intent_str = [
            intent
            for intent in self.intent_str
            if intent in SLU_INTENT_STR["SpeechMassive"]
        ]

        self.config = config
        partitions = config.partitions

        assert config.task in ["ASR", "SLU"], NotImplementedError(
            "Only ASR ans SLU task is supported for Slurp dataset"
        )

        datasets = defaultdict(list)
        (
            self.train_dataset,
            self.test_dataset,
            self.eval_dataset,
            self.few_shot_dataset,
        ) = (
            None,
            None,
            None,
            None,
        )

        self.few_shot_split = "train"
        self.few_shot_samples = 115

        preprocess_fn = getattr(self, f"preprocess_{config.task}")

        assert (
            len(config.languages) == 1 and config.languages[0] in "en"
        ), ValueError("Only English language is supported for Slurp dataset")

        if self.config.task == "SLU":
            logging.info(f"Loading few_shot dataset for SLU")
            dataset = load_dataset(
                "parquet",
                data_files={
                    self.few_shot_split: f"{config.datapath}/slurp.{self.few_shot_split}.parquet",
                },
                split=self.few_shot_split,
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if rebuild_cache
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )

            dataset = dataset.shuffle(seed=42).select(
                range(self.few_shot_samples)
            )

            # map the few_shot dataset
            dataset = dataset.map(
                lambda example: self.preprocess_few_shot(example),
                batched=False,
            )

            datasets["few_shot"].append(dataset)

        if "few_shot" in datasets and len(datasets["few_shot"]):
            print(f"Concatenating few_shot dataset")
            if len(dataset) == 1:
                self.few_shot_dataset = datasets["few_shot"][0]
            elif len(dataset):
                self.few_shot_dataset = concatenate_datasets(
                    datasets["few_shot"]
                )

        for split, info in partitions.items():
            logging.info(f"Loading {split} dataset")
            # FIXME
            dataset = load_dataset(
                "parquet",
                data_files={
                    f"{split}": f"{config.datapath}/slurp.{split}.parquet",
                },
                split=f"{split}[{info['amount']}]",
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if rebuild_cache
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )

            min_duration = info["min_duration"]
            max_duration = info["max_duration"]

            if min_duration is not None and max_duration is not None:
                dataset = dataset.filter(
                    lambda example: min_duration
                    <= self.get_duration(example)
                    <= max_duration
                )
                logging.info(
                    f"Filtering dataset with min_duration: {min_duration} and max_duration: {max_duration}"
                )
            elif min_duration is not None:
                dataset = dataset.filter(
                    lambda example: min_duration <= self.get_duration(example)
                )
                logging.info(
                    f"Filtering dataset with min_duration: {min_duration}"
                )
            elif max_duration is not None:
                dataset = dataset.filter(
                    lambda example: self.get_duration(example) <= max_duration
                )
                logging.info(
                    f"Filtering dataset with max_duration: {max_duration}"
                )

            dataset = dataset.map(
                lambda example: preprocess_fn(example),
                batched=False,
            )
            datasets[info["destination"]].append(dataset)

        for destination, dataset in datasets.items():
            print(f"Concatenating {destination} dataset")
            if len(dataset) == 1:
                setattr(self, f"{destination}_dataset", dataset[0])
            elif len(dataset):
                setattr(
                    self,
                    f"{destination}_dataset",
                    concatenate_datasets(dataset),
                )

    def __len__(self):
        return len(self.train_dataset)

    def get_duration(self, example):
        return example["duration"]

    def preprocess_few_shot(self, example):
        source = "en"
        example["source_language"] = source
        example["intent_str"] = example["intent"]
        example["annot_utt"] = self.get_annot_utt(example)

        slu_inputs_text = self.config.INPUTS_TEXT_LIST[self.config.task]
        output_format_version = slu_inputs_text.get("output_format_version", 1)

        example["output"] = self.get_formatted_output(
            example, output_format_version
        )

        return example

    def preprocess_ASR(self, example):
        source = "en"
        example["source_language"] = source
        example["target_language"] = None
        example["task"] = self.config.task
        example["transcription"] = example["transcript"]

        prompt_language = "en"

        question, answer = "", ""

        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][prompt_language]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][
                    prompt_language
                ]
            )
        example["conversations"] = [
            {
                "from": "human",
                "value": DEFAULT_AUDIO_INPUT_START_TOKEN
                + DEFAULT_AUDIO_INPUT_END_TOKEN
                + "\n"
                + question,
            },
            {"from": "gpt", "value": answer + example["transcript"]},
        ]

        example["audio_input"] = example["audio"]

        for k in list(example.keys()):
            if k not in [
                "audio_input",
                "conversations",
                "source_language",
                "target_language",
                "task",
                "transcription",
            ]:
                del example[k]

        return example

    def preprocess_SLU(self, example):
        source = "en"
        example["source_language"] = source
        example["target_language"] = None
        example["task"] = self.config.task
        example["transcription"] = example["transcript"]

        prompt, answer = "", ""

        if self.config.INPUTS_TEXT_LIST is None:
            raise ValueError("INPUTS_TEXT_LIST is required for SLU task")

        slu_inputs_text = self.config.INPUTS_TEXT_LIST[self.config.task]
        output_format_version = slu_inputs_text.get("output_format_version", 1)

        # Add task explanation
        prompt += (
            "### Task Explanation\n"
            + slu_inputs_text["task_explanation"]
            + "\n\n"
        )

        # Add intent strings and slot types (if needed for the task)
        if self.intent_str:
            prompt += "#### Intent Strings (intent_str):\n"
            prompt += (
                "\n".join(f"- {intent}" for intent in self.intent_str) + "\n\n"
            )

        if self.slot_type:
            prompt += "#### Slot Types (slot_type):\n"
            prompt += (
                "\n".join(f"- {slot}" for slot in self.slot_type) + "\n\n"
            )

        if "input_format" in slu_inputs_text:
            prompt += (
                "#### Input Format:\n"
                + slu_inputs_text["input_format"]
                + "\n\n"
            )

        if "output_format" in slu_inputs_text:
            prompt += (
                "#### Output Format:\n"
                + slu_inputs_text["output_format"]
                + "\n\n"
            )

        example["annot_utt"] = self.get_annot_utt(example)
        # Add example input and output format
        for i in range(SLU_N_FEW_SHOT):
            # choose a random sample from the few_shot dataset
            few_shot_sample = random.choice(self.few_shot_dataset)

            # check that few_shot_sample is different from the current example
            while few_shot_sample["annot_utt"] == example["annot_utt"]:
                few_shot_sample = random.choice(self.few_shot_dataset)

            input = (
                "language: "
                + LANGUAGES_CODE_NAME["en"][few_shot_sample["source_language"]]
            )

            output = few_shot_sample["output"]

            prompt += f"#### Example Input {i}:\n{input}\n\n"
            prompt += f"#### Example Output {i}:\n{output}\n\n"

        # Add instructions
        if "instructions" in slu_inputs_text:
            prompt += "Instructions:\n"
            prompt += (
                "\n".join(
                    f"{i+1}. {instruction}"
                    for i, instruction in enumerate(
                        slu_inputs_text["instructions"]
                    )
                )
                + "\n"
            )

        if not source in LANGUAGES_CODE_NAME["en"]:
            raise ValueError(
                f"Please add {source} to LANGUAGES_CODE_NAME['en']"
            )

        language = LANGUAGES_CODE_NAME["en"][source]
        prompt += "### Input\nlanguage: " + language + "\n"

        gt = self.get_formatted_output(example, output_format_version)

        example["conversations"] = [
            {
                "from": "human",
                "value": DEFAULT_AUDIO_INPUT_START_TOKEN
                + DEFAULT_AUDIO_INPUT_END_TOKEN
                + "\n"
                + prompt,
            },
            {"from": "gpt", "value": str(gt)},
        ]

        example["audio_input"] = example["audio"]

        for k in list(example.keys()):
            if k not in [
                "audio_input",
                "conversations",
                "source_language",
                "target_language",
                "task",
                "transcription",
            ]:
                del example[k]

        return example

    def get_annot_utt(self, example):
        # example["slots:"] = [ "email_folder=inbox" ]
        # create annot_utt from slots, replacing slot values with [slot_type : slot_value]
        annot_utt = example["transcript"]
        count_slots_values = {}

        def replace_nth_occurrence(string, old, new, n):
            if n == 0:
                return string
            parts = string.split(old, n)
            if len(parts) <= n:
                return string
            return old.join(parts[:-1]) + new + parts[-1]

        for slot in example["slots:"]:
            slot_type, slot_value = slot.split("=")
            slot_value = slot_value.lower()
            if slot_value not in annot_utt:
                # some slot values have a space before 's, even though it is not present in the transcript
                # print(f"Slot value `{slot_value}` not found in `{annot_utt}`")
                slot_value = slot_value.replace(" 's", "'s")
                # print(f"Trying to remove space before 's -> `{slot_value}`")
                if slot_value not in annot_utt:
                    print(
                        f"NOT FOUND: Slot value `{slot_value}` not found in `{annot_utt}` after removing space before 's"
                    )
                    continue
                else:
                    print(
                        f"FOUND: Slot value `{slot_value}` found in transcript after removing space before 's"
                    )

            count_slots_values[slot_value] = (
                count_slots_values.get(slot_value, 0) + 1
            )
            # replace count_slots_values[slot_value]th occurence of slot_value with [slot_type : slot_value]
            annot_utt = replace_nth_occurrence(
                annot_utt,
                slot_value,
                f"[{slot_type} : {slot_value}]",
                count_slots_values[slot_value],
            )

        return annot_utt

    def get_formatted_output(self, example, version):
        if version == 1:
            output = dict()
            output["intent_str"] = example["intent"]
            output["annot_utt"] = example["annot_utt"]
            return str(output)
        elif version == 2:
            output = f"intent_str: {example['intent']}"
            for slot in example["slots:"]:
                slot_type, slot_value = slot.split("=")
                output += f", {slot_type}: {slot_value}"
            return output
        else:
            raise ValueError("Invalid version")
