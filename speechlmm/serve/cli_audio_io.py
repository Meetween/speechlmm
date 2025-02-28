import argparse
import logging
import os
import random
import re
import string
from io import BytesIO

import requests
import torch
import torchaudio
from num2words import num2words
from PIL import Image
from transformers import TextStreamer

from speechlmm.constants import (
    DEFAULT_AUDIO_CONDITION_END_TOKEN,
    DEFAULT_AUDIO_CONDITION_START_TOKEN,
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
)
from speechlmm.conversation import SeparatorStyle, conv_templates
from speechlmm.dataset.config import (
    LANGUAGES_CODE_NAME,
    SOURCE_LANGUAGE_PLACEHOLDER,
    TARGET_LANGUAGE_PLACEHOLDER,
)
from speechlmm.mm_utils import monify_and_resample_audio
from speechlmm.model.builder import load_pretrained_model
from speechlmm.utils import torch_dtype_from_str

# Constants for hardcoded prompts and paths
TTS_PROMPT = "Read out loud the following text: "
ASR_PROMPT = (
    DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
    + "\n"
    + "Transcribe the audio."
)
CONDITION_PROMPT = (
    DEFAULT_AUDIO_CONDITION_START_TOKEN + DEFAULT_AUDIO_CONDITION_END_TOKEN
)
S2ST_PROMPT_LANGUAGE = "en"
S2ST_PROMPT = (
    DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
    + "\n"
    + "Can you translate this audio from {source_language} into {target_language}?"
)
S2ST_PROMPT_W_CLONING = (
    DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
    + "\n"
    + DEFAULT_AUDIO_CONDITION_START_TOKEN
    + DEFAULT_AUDIO_CONDITION_END_TOKEN
    + "\n"
    + "Can you translate this audio from {source_language} into {target_language}?"
)
DEFAULT_AUDIO_PATH = (
    f"{os.getenv('SCRATCH', '.')}/libritts_r_conditioning/libritts_r_1988.wav"
)
DEFAULT_SPEAKER_DESCRIPTION_PREFIX = "Speech description: "
DEFAULT_SPEAKER_DESCRIPTION = (
    DEFAULT_SPEAKER_DESCRIPTION_PREFIX
    + "A man speaks with a very expressive and animated tone, his voice "
    "sounding clear and very close-up. He delivers his speech slightly faster "
    "than usual, with a moderate pitch."
)

SUPPORTED_SOURCE_LANGUAGES = ["de", "es", "it"]
SUPPORTED_TARGET_LANGUAGES = ["en", "de"]


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_audio(audio_path):
    """
    Load audio from a local file path or URL using torchaudio.

    Parameters:
    - audio_path (str): The local path or URL of the audio file.

    Returns:
    - waveform (torch.Tensor): 1D tensor representing the audio waveform.
    - sample_rate (int): The sample rate of the audio.
    """

    try:
        # Load audio using torchaudio
        audio, sr = torchaudio.load(audio_path, normalize=True)
        return (audio, sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


def load_image_tensor(image, model, processor, model_config):
    image_tensor = process_images([image], processor, model_config)
    if type(image_tensor) is list:
        image_tensor = [
            image.to(model.device, dtype=torch.float16)
            for image in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    return image_tensor


def load_audio_into_tensor(audio_path, target_sr=None):
    audio, orig_sr = torchaudio.load(
        audio_path, normalize=True, channels_first=True
    )
    return monify_and_resample_audio(audio, orig_sr, target_sr)


def load_audio_for_conditions(path, target_sr):
    if path:
        try:
            print(f"Loading condition audio from {path}")
            condition_audio, condition_sr = load_audio_into_tensor(
                path, target_sr=target_sr
            )
            condition_audio = condition_audio.to(dtype=torch.float16)
            return [(condition_audio, condition_sr)]
        except Exception as e:
            raise Exception("Error loading condition audio.") from e
    return None


def load_audio_for_asr(path, target_sr):
    if path:
        try:
            print(f"Loading ASR audio from {path}")
            asr_audio, asr_sr = load_audio_into_tensor(
                path, target_sr=target_sr
            )
            asr_audio = asr_audio.to(dtype=torch.float16)
            return [(asr_audio, asr_sr)]
        except Exception as e:
            raise Exception("Error loading ASR audio.") from e
    return None


import re


def construct_prompt(
    task,
    user_inp,
    tts_inp,
    speaker_description=None,
    source_language=None,
    target_language=None,
):
    if task == "tts" and tts_inp:
        tts_inp = re.sub(
            r"\d+(\.\d+)?",
            lambda x: num2words(x.group(), lang=target_language),
            tts_inp,
        )
        # TTS task with user input and TTS input
        condition_prompt = (
            speaker_description if speaker_description else CONDITION_PROMPT
        )
        prompt = (
            f"{user_inp}\n{TTS_PROMPT}{tts_inp}\n{condition_prompt}".strip()
            if user_inp
            else f"{TTS_PROMPT}{tts_inp}\n{condition_prompt}".strip()
        )
    elif task == "asr":
        prompt = ASR_PROMPT.strip()
    elif task == "s2st" or task == "s2st_cloning":
        if "cloning" in task:
            s2st_prompt = S2ST_PROMPT_W_CLONING.format(
                source_language=source_language,
                target_language=target_language,
                speaker_description=speaker_description,
            )
        else:
            s2st_prompt = S2ST_PROMPT.format(
                source_language=source_language,
                target_language=target_language,
            )
        prompt = (
            f"{user_inp}\n{s2st_prompt}".strip()
            if user_inp
            else f"{s2st_prompt}".strip()
        )
    elif task == "audio_inp":
        prompt = (
            DEFAULT_AUDIO_INPUT_START_TOKEN
            + DEFAULT_AUDIO_INPUT_END_TOKEN
            + "\n"
            + user_inp
        )
    else:
        # Only user input
        prompt = user_inp.strip()
    return prompt


def process_single_task(conv, args, model, tokenizer):
    """
    Process a single task from the user.

    Returns:
        conv: Updated conversation object.
        continue_loop (bool): Whether to continue the main loop.
    """
    # User inputs
    tts_inp = ""
    asr_inp_audio_path = ""
    condition_audio_path = ""
    user_inp = ""
    condition_audios = None
    input_audios = None
    speaker_description = None
    source_language = None
    target_language = None
    task = None

    # Create temporary directory for audio files
    tmp_dir = f"{os.getenv('SCRATCH', '.')}/generated"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        task = (
            input(
                "TASK [asr, tts, s2st, s2st_cloning, audio_inp, empty string for none]: "
            )
            .strip()
            .lower()
            or None
        )

        # Gather necessary user inputs
        if task == "tts":
            tts_inp = input("TTS INPUT: ").strip()
            # ask for target language
            target_language = (
                input(
                    f"Target language [{', '.join(SUPPORTED_TARGET_LANGUAGES)}]: "
                )
                .strip()
                .lower()
            )
            if args.caption_only:
                speaker_description_input = input(
                    "SPEAKER DESCRIPTION: (leave empty for default): "
                ).strip()
                speaker_description = (
                    DEFAULT_SPEAKER_DESCRIPTION_PREFIX
                    + speaker_description_input
                    if speaker_description_input
                    else DEFAULT_SPEAKER_DESCRIPTION
                )
            else:
                condition_audio_path = (
                    input(
                        "CONDITION AUDIO PATH (leave empty for default): "
                    ).strip()
                    or DEFAULT_AUDIO_PATH
                )
        elif task == "asr":
            asr_inp_audio_path = (
                input("ASR audio path (leave empty for default): ").strip()
                or DEFAULT_AUDIO_PATH
            )
        elif task == "s2st" or task == "s2st_cloning":
            # Ask for source and target languages
            source_language = (
                input(
                    f"Source language [{', '.join(SUPPORTED_SOURCE_LANGUAGES)}]: "
                )
                .strip()
                .lower()
            )

            target_language = input(f"Target language [en]: ").strip().lower()

            source_language = LANGUAGES_CODE_NAME[S2ST_PROMPT_LANGUAGE][
                source_language
            ]
            target_language = LANGUAGES_CODE_NAME[S2ST_PROMPT_LANGUAGE][
                target_language
            ]

            asr_inp_audio_path = (
                input("Input audio path (leave empty for default): ").strip()
                or DEFAULT_AUDIO_PATH
            )
            if "cloning" in task:
                condition_audio_path = (
                    input(
                        "CONDITION AUDIO PATH (leave empty for default): "
                    ).strip()
                    or DEFAULT_AUDIO_PATH
                )
        elif task == "audio_inp":
            asr_inp_audio_path = input("AUDIO INPUT PATH: ").strip()
            user_inp = input("USER INPUT: ").strip()
        else:
            user_inp = input("USER INPUT: ").strip()

        if user_inp.lower() == "<reset>":
            print("Resetting conversation...")
            conv = conv_templates[args.conv_mode].copy()
            return conv, True  # Reset conversation, continue loop
    except EOFError:
        print("EOFError encountered, exiting loop...")
        return conv, False  # Exit loop

    # Load audios if specified
    if task == "tts" and not args.caption_only:
        condition_audios = load_audio_for_conditions(
            condition_audio_path, model.codec_encoder.input_sampling_rate
        )
    elif task == "asr" or task == "audio_inp":
        input_audios = load_audio_for_asr(
            asr_inp_audio_path, model.audio_encoder.input_sampling_rate
        )
    elif task == "s2st" or task == "s2st_cloning":
        input_audios = load_audio_for_asr(
            asr_inp_audio_path, model.audio_encoder.input_sampling_rate
        )
        if "cloning" in task:
            condition_audios = load_audio_for_conditions(
                condition_audio_path, model.codec_encoder.input_sampling_rate
            )

    # Construct prompt based on task and inputs
    prompt = construct_prompt(
        task,
        user_inp,
        tts_inp,
        speaker_description if args.caption_only else None,
        source_language=source_language,
        target_language=target_language,
    )

    if args.conv_mode == "llama_3_1_base":
        # Tokenize with only BOS token (no EOS)
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,  # Disable automatic special tokens
        ).input_ids
        # Manually prepend BOS token
        # input_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), input_ids], dim=0)
        bos_tensor = torch.full(
            (input_ids.shape[0], 1),
            tokenizer.bos_token_id,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        prompt_full = tokenizer.decode(input_ids[0])

    else:
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        prompt_full = conv.get_prompt()
        print(f"prompt: {prompt_full}")

        input_ids = tokenizer(prompt_full, return_tensors="pt").input_ids.to(
            model.device
        )

    print(f"input_ids: {input_ids}")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    if args.debug:
        streamer = None

    with torch.inference_mode():
        kwargs = {}
        if task == "tts":
            kwargs.update(
                {
                    "tts_input": tts_inp,
                    "force_text_tokens": args.force_text_tokens,
                }
            )
        elif task == "s2st":
            kwargs.update(
                {"tts_input": "", "force_text_tokens": args.force_text_tokens}
            )

        if condition_audios is not None:
            print(f"Using condition audio")

        import time

        start = time.time()

        print("Generating...")
        output = model.generate(
            input_ids,
            images=None,
            image_sizes=None,
            audios=input_audios,
            condition_audios=condition_audios,  # condition_audios,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            tts_max_len=args.tts_max_len,
            streamer=streamer,
            use_cache=True,
            **kwargs,
        )

        if len(output) > 1:  #
            audio_values, output_ids = output
            # Save the generated audio
            if audio_values is not None:
                for audio_value in audio_values:
                    audio_id = "".join(
                        random.choices(
                            string.ascii_lowercase + string.digits, k=7
                        )
                    )
                    filename = f"{tmp_dir}/{audio_id}.wav"
                    torchaudio.save(
                        filename,
                        audio_value.to(dtype=torch.float32).detach().cpu(),
                        24000,
                    )

                    print("Saved at ", filename)
        else:
            output_ids = output[0]
            print("----------")

        print("Inference time: ", time.time() - start)

    try:
        output_text = tokenizer.decode(output_ids).strip()
    except Exception as e:
        print("Error decoding output: ", e)
        output_text = None

    print("\n", {"prompt": prompt_full, "outputs": output_text}, "\n")
    conv = conv_templates[args.conv_mode].copy()
    return conv, True  # Continue loop


def main(args):
    model = load_pretrained_model(
        args.model_path,
        attn_implementation=args.attn_implementation,
        device=args.device,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = model.text_decoder.tokenizer

    model.half()

    conv_mode = model.text_decoder.conversation_version
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        logging.warning(
            f"The auto-inferred conversation mode is {conv_mode}, while "
            f"`--conv-mode` is {args.conv_mode}. Using the latter."
        )
    else:
        args.conv_mode = conv_mode

    logging.info(f"Using conversation mode '{args.conv_mode}'")

    conv = conv_templates[args.conv_mode].copy()

    # Main interaction loop
    while True:
        conv, continue_loop = process_single_task(conv, args, model, tokenizer)
        if not continue_loop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument(
        "--torch-dtype", type=torch_dtype_from_str, default=None
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--caption-only", action="store_true")
    parser.add_argument("--tts-max-len", type=int, default=5)
    parser.add_argument("--force-text-tokens", action="store_true")
    args = parser.parse_args()
    main(args)


# python speechlmm/serve/cli_audio_io.py \
#     --model-path "$CHECKPOINTS_HOME/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_speaker_description/checkpoint-7500" \
#     --conv-mode "llama_3_1" \
#     --tts-max-len 30 \
#     --caption-only
