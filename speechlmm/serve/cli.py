import argparse
import json
import logging
import math
import os

import torch
import torchaudio
from transformers import TextStreamer

from speechlmm.constants import (
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
    DEFAULT_VIDEO_INPUT_END_TOKEN,
    DEFAULT_VIDEO_INPUT_START_TOKEN,
)
from speechlmm.conversation import conv_templates
from speechlmm.dataset.config import LANGUAGES_CODE_NAME
from speechlmm.mm_utils import monify_and_resample_audio
from speechlmm.model.builder import load_pretrained_model
from speechlmm.serve.slu_intent_only_utils import get_intents, get_slot_types
from speechlmm.utils import disable_torch_init, torch_dtype_from_str

# Load the inputs text list from JSON file
INPUTS_TEXT_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "INPUTS_TEXT_LIST.json"
)
with open(INPUTS_TEXT_LIST_PATH, "r") as f:
    INPUTS_TEXT_LIST = json.load(f)

ALLOWED_TASKS = ["ASR", "ST", "SLU", "SSUM", "VSR", "SQA"]
# FIXME SLU must be SLU INTENT ONLY


def load_audio_into_tensor(audio_path, target_sr=None):
    if "'" in audio_path:
        audio_path = audio_path.replace("'", "")
    audio, orig_sr = torchaudio.load(
        audio_path, normalize=True, channels_first=True
    )
    return monify_and_resample_audio(audio, orig_sr, target_sr)


def load_video_into_tensor(video_path, dtype, target_sr=None):
    import decord
    import torchvision

    video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))

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

    video = video.to(torch.float32)
    video = torchvision.transforms.functional.normalize(video, mean=0, std=255)
    video = torchvision.transforms.functional.center_crop(
        video, output_size=88
    )
    video = torchvision.transforms.functional.rgb_to_grayscale(
        video, num_output_channels=1
    )
    video = torchvision.transforms.functional.normalize(
        video, mean=0.421, std=0.165
    )
    return video.to(dtype=dtype), target_sr


def select_task():
    language = "en"
    source_language = None
    target_language = None
    user_TASK = input(
        f"Select a task from the following {ALLOWED_TASKS} or just leave empty to have a chat: "
    ).upper()
    assert (
        user_TASK == "" or user_TASK in ALLOWED_TASKS
    ), f"You have prompted an invalid task: {user_TASK}. Please select one of the following: {ALLOWED_TASKS} or just leave empty to have a chat."

    if user_TASK == "SLU":
        user_TASK = "SLU_INTENT_ONLY"

    if user_TASK == "ST":
        source_language = input(
            f"Source language: ({list(INPUTS_TEXT_LIST[user_TASK].keys())}): "
        )
        if source_language not in list(INPUTS_TEXT_LIST[user_TASK].keys()):
            raise ValueError(
                f"Invalid language: {source_language}. Please select one of the following: {list(INPUTS_TEXT_LIST[user_TASK].keys())}"
            )
        target_language = input(
            f"Target language: ({list(INPUTS_TEXT_LIST[user_TASK].keys())}): "
        )
        if target_language not in list(INPUTS_TEXT_LIST[user_TASK].keys()):
            raise ValueError(
                f"Invalid language: {target_language}. Please select one of the following: {list(INPUTS_TEXT_LIST[user_TASK].keys())}"
            )
        if source_language == target_language:
            raise ValueError(
                f"Source and target languages cannot be the same: {source_language}"
            )
        language = target_language
    elif user_TASK != "":
        language = input(
            f"Language: ({list(INPUTS_TEXT_LIST[user_TASK].keys())}). Default en. Please note that when the task is not ST, the language of the audio is automatically inferred. "
        ).strip()
        if language != "" and language not in list(
            INPUTS_TEXT_LIST[user_TASK].keys()
        ):
            raise ValueError(
                f"Invalid language: {language}. Please select one of the following: {list(INPUTS_TEXT_LIST[user_TASK].keys())}"
            )
        elif language == "":
            language = "en"
    if user_TASK not in ["SQA", ""]:
        print("")
        logging.info(
            "Conversation history is not enabled when not in SQA or chat mode. Please note that after the first utterance, the conversation history will be reset."
        )
    return user_TASK, language, source_language, target_language


def get_SLU_INTENT_ONLY_prompt(language):
    slu_task_specification = INPUTS_TEXT_LIST["SLU_INTENT_ONLY"][language]
    prompt = slu_task_specification["task_explanation"] + "\n\n"
    intents = get_intents()
    if len(intents) > 0:
        prompt += "Intent strings:\n"
        prompt += "\n".join(f"- {intent}" for intent in intents) + "\n\n"
    if "input_format" in slu_task_specification:
        prompt += (
            "Input format:\n" + slu_task_specification["input_format"] + "\n\n"
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
    promp_language = LANGUAGES_CODE_NAME["en"][language]
    input_ = (
        prompt
        + "Spoken utterance: "
        + DEFAULT_AUDIO_INPUT_START_TOKEN
        + DEFAULT_AUDIO_INPUT_END_TOKEN
        + "\n"
        + f"Language of the spoken utterance: {promp_language}\n"
    )
    return input_


def main(args):
    audio, video = None, None
    model = load_pretrained_model(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    # handle pad token
    model.text_decoder.model.generation_config.pad_token_id = (
        model.text_decoder.tokenizer.pad_token_id
    )

    conv_mode = model.text_decoder.conversation_version
    conv = conv_templates[conv_mode].copy()

    user_TASK, language, source_language, target_language = select_task()
    while True:
        if user_TASK == "":
            inp = input("USER: ")
            conv.append_message(conv.roles[0], inp)
        elif user_TASK == "VSR":
            video_path = input("Enter the path to the video file: ")
            video, sr = load_video_into_tensor(video_path, dtype=model.dtype)
            conv = conv_templates[conv_mode].copy()

            inp = (
                DEFAULT_VIDEO_INPUT_START_TOKEN
                + DEFAULT_VIDEO_INPUT_END_TOKEN
                + "\n"
                + INPUTS_TEXT_LIST[user_TASK][language][0]
            )
            conv.append_message(conv.roles[0], inp)
        else:
            if audio is None or user_TASK != "SQA":
                audio_path = input("Enter the path to the audio file: ")
                audio, sr = load_audio_into_tensor(
                    audio_path,
                    target_sr=model.audio_encoder.input_sampling_rate,
                )
            if audio is not None:
                conv = conv_templates[conv_mode].copy()

            logging.debug("adding audio token")
            if user_TASK != "SLU_INTENT_ONLY":
                inp = (
                    DEFAULT_AUDIO_INPUT_START_TOKEN
                    + DEFAULT_AUDIO_INPUT_END_TOKEN
                    + "\n"
                    + INPUTS_TEXT_LIST[user_TASK][language][0]
                )
                if user_TASK == "ST":
                    inp = inp.replace(
                        "<source-language>",
                        LANGUAGES_CODE_NAME[language][source_language],
                    )
                    inp = inp.replace(
                        "<target-language>",
                        LANGUAGES_CODE_NAME[language][target_language],
                    )
                elif user_TASK == "SQA":
                    question = input("Enter the question: ")
                    inp = inp.replace("<question>", question)
            elif user_TASK == "SLU_INTENT_ONLY":
                inp = get_SLU_INTENT_ONLY_prompt(language)
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = model.text_decoder.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(model.device)

        streamer = TextStreamer(
            model.text_decoder.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        print("")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                audios=[(audio, sr)] if audio is not None else None,
                videos=[(video, sr)] if video is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
            )

        outputs = model.text_decoder.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if user_TASK not in ["SQA", ""]:
            conv = conv_templates[conv_mode].copy()
            repeat = input(
                f"\nDo you want continuing the {user_TASK} task? (y/n) yes by default: "
            )
            if repeat == "" or repeat.startswith("y"):
                continue
            else:
                user_TASK, language, source_language, target_language = (
                    select_task()
                )

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument(
        "--attn-implementation", type=str, default="flash_attention_2"
    )
    parser.add_argument(
        "--torch-dtype", type=torch_dtype_from_str, default=None
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)

# sample usage:
# python /serve/cli.py --model-path "$CHECKPOINTS_HOME/speechlmm-v1/checkpoints/v1/l/checkpoint-1500"
