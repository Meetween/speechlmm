import json
import os
from typing import Optional

from speechlmm.constants import (
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
)
from speechlmm.conversation import conv_templates
from speechlmm.dataset.config import LANGUAGES_CODE_NAME
from speechlmm.serve.slu_intent_only_utils import get_intents

INPUTS_TEXT_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "INPUTS_TEXT_LIST.json"
)
with open(INPUTS_TEXT_LIST_PATH, "r") as f:
    INPUTS_TEXT_LIST = json.load(f)

ALLOWED_TASKS = ["ASR", "ST", "SLU", "SSUM", "VSR", "SQA"]


class PromptBuilder:
    def __init__(
        self,
        task: str,
        conv_mode: str,
        tokenizer,
        language: str = "en",
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ):
        self.task = task
        self.language = language
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.source_language = source_language
        self.target_language = target_language

        if self.task not in ALLOWED_TASKS:
            raise ValueError(
                f"Invalid task: {self.task}. Please select one of the following: {ALLOWED_TASKS}"
            )
        if self.task == "ST":
            assert (
                self.source_language is not None
                and self.target_language is not None
            ), "Source and target languages must be provided for ST task"
            assert (
                self.source_language != self.target_language
            ), "Source and target languages must be different for ST task"
        if self.task == "SLU":
            self.task = "SLU_INTENT_ONLY"
            assert (
                self.language == "en"
            ), "Language must be English for SLU task"

    def reset_conv(self):
        self.conv = conv_templates[self.conv_mode].copy()

    def _get_SLU_INTENT_ONLY_prompt(self):
        slu_task_specification = INPUTS_TEXT_LIST["SLU_INTENT_ONLY"][
            self.language
        ]
        prompt = slu_task_specification["task_explanation"] + "\n\n"
        intents = get_intents()
        if len(intents) > 0:
            prompt += "Intent strings:\n"
            prompt += "\n".join(f"- {intent}" for intent in intents) + "\n\n"
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
        promp_language = LANGUAGES_CODE_NAME["en"][self.language]
        input_ = (
            prompt
            + "Spoken utterance: "
            + DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + f"Language of the spoken utterance: {promp_language}\n"
        )
        return input_

    def get_prompt(self, question: Optional[str] = None):
        self.reset_conv()
        if self.task == "SLU_INTENT_ONLY":
            return self._get_SLU_INTENT_ONLY_prompt()
        else:
            inp = (
                DEFAULT_AUDIO_INPUT_START_TOKEN
                + DEFAULT_AUDIO_INPUT_END_TOKEN
                + "\n"
                + INPUTS_TEXT_LIST[self.task][self.language][0]
            )
            if self.task == "ST":
                inp = inp.replace(
                    "<source-language>",
                    LANGUAGES_CODE_NAME[self.language][self.source_language],
                )
                inp = inp.replace(
                    "<target-language>",
                    LANGUAGES_CODE_NAME[self.language][self.target_language],
                )
            elif self.task == "SQA":
                assert (
                    question is not None
                ), "Question must be provided for SQA task"
                inp = inp.replace("<question>", question)
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        return prompt
