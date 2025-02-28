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
from tqdm import tqdm
from transformers import TextStreamer

from speechlmm.arguments import DataArguments
from speechlmm.constants import (
    DEFAULT_AUDIO_CONDITION_END_TOKEN,
    DEFAULT_AUDIO_CONDITION_START_TOKEN,
    DEFAULT_AUDIO_INPUT_END_TOKEN,
    DEFAULT_AUDIO_INPUT_START_TOKEN,
)
from speechlmm.conversation import SeparatorStyle, conv_templates
from speechlmm.dataset.config import LANGUAGES_CODE_NAME
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.dataset.speechlmm_dataset import (
    SpeechLmmDataset,
    SpeechLmmInferenceDataset,
)
from speechlmm.mm_utils import monify_and_resample_audio
from speechlmm.model.builder import load_pretrained_model
from speechlmm.utils import disable_torch_init

# Constants for hardcoded prompts and paths
TTS_PROMPT = "Read out loud the following text: "
ASR_PROMPT = (
    "Repeat after me: "
    + DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
)
CONDITION_PROMPT = (
    DEFAULT_AUDIO_CONDITION_START_TOKEN + DEFAULT_AUDIO_CONDITION_END_TOKEN
)
S2ST_PROMPT_TEMPLATE = (
    DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
    + "\n"
    + "Can you translate this audio from {source_language} into {target_language}?"
)
S2ST_CLONING_PROMPT_TEMPLATE = (
    DEFAULT_AUDIO_INPUT_START_TOKEN
    + DEFAULT_AUDIO_INPUT_END_TOKEN
    + "\n"
    + DEFAULT_AUDIO_CONDITION_START_TOKEN
    + DEFAULT_AUDIO_CONDITION_END_TOKEN
    + "\n"
    + "Can you translate this audio from {source_language} into {target_language}?"
)
DEFAULT_AUDIO_PATH = f"{os.getenv('SCRATCH', '.')}/audio_cli/male.wav"
DEFAULT_SPEAKER_DESCRIPTION_PREFIX = "Speech description: "
DEFAULT_SPEAKER_DESCRIPTION = (
    DEFAULT_SPEAKER_DESCRIPTION_PREFIX
    + "A person speaks with a very expressive and animated tone, the voice "
    "sounding clear and very close-up. They deliver speech slightly faster "
    "than usual, with a moderate pitch."
)

SUPPORTED_SOURCE_LANGUAGES = ["de", "es", "it", "fr"]
SUPPORTED_TARGET_LANGUAGE = "en"
SUPPORTED_TTS_LANGUAGES = ["en", "de"]


def load_audio_into_tensor(audio_path, target_sr=None):
    audio, orig_sr = torchaudio.load(
        audio_path, normalize=True, channels_first=True
    )
    return monify_and_resample_audio(audio, orig_sr, target_sr)


def get_inference_dataloader(config_path, model):
    data_args = DataArguments(
        codec_sampling_rate=model.codec_encoder.input_sampling_rate,
        codec_frame_rate=model.codec_encoder.output_sampling_rate,
        data_config_path=config_path,
        is_multimodal=True,
        dataloader_debug=False,
        group_dataset_by_task={
            "train": False,
            "eval": False,
            "test": False,
        },
        organize_eval_dataset_per_task=True,
        align_text_to_audio="text_pad_epad",
        align_with_whisper=False,
        restore_punctuation_and_spaces=True,
        rebuild_dataset_cache=False,
        num_proc_for_preprocessing=32,
    )
    datasets = DatasetsWrapper(data_args)
    print("Test dataset: ", datasets.test_dataset)
    dataset = SpeechLmmInferenceDataset(
        dataset=datasets.test_dataset,
        data_args=data_args,
    )
    return dataset.get_data_loader(batch_size=1, shuffle=False)


def load_audio_for_conditions(path, target_sr):
    if path:
        try:
            print(f"Loading conditioning audio from {path}")
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
            print(f"Loading input audio from {path}")
            asr_audio, asr_sr = load_audio_into_tensor(
                path, target_sr=target_sr
            )
            asr_audio = asr_audio.to(dtype=torch.float16)
            return [(asr_audio, asr_sr)]
        except Exception as e:
            raise Exception("Error loading input audio.") from e
    return None


def load_audio_from_dict(dict, audio_key, target_sr):
    audio = dict[audio_key]
    sr = dict[audio_key + "_sr"]
    audio, sr = monify_and_resample_audio(audio.unsqueeze(0), sr, target_sr)
    audio = audio.to(dtype=torch.float16)
    return [(audio, sr)]


def generate_filename_summary(sentence):
    """
    Generate a 1-2 word non-ambiguous summary of the sentence for the filename.
    """
    import re

    # Remove punctuation and make lowercase
    clean_sentence = re.sub(r"[^\w\s]", "", sentence).lower()
    words = clean_sentence.split()
    # Use the first two significant words
    significant_words = [word for word in words if len(word) > 2]
    summary = "_".join(significant_words[:2]) if significant_words else "audio"
    return summary


def increment_path(output_path):
    if os.path.exists(output_path):
        dir, filename = os.path.split(output_path)
        filename, ext = os.path.splitext(filename)
        idx = 1
        while os.path.exists(output_path):
            output_path = os.path.join(dir, f"{filename}_{idx}.{ext}")
            idx += 1
    return output_path


def find_audio_files(directory):
    """
    Recursively find all .wav files in a directory and return their paths.
    """
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav") or file.lower().endswith(".mp3"):
                # Get relative path from directory
                rel_dir = os.path.relpath(root, directory)
                if rel_dir == ".":
                    rel_dir = ""
                else:
                    rel_dir = rel_dir.replace(os.sep, "-")
                # Build the file path
                file_path = os.path.join(root, file)
                # Build the filename with subdirectory names using '-'
                if rel_dir:
                    filename = f"{rel_dir}-{file}"
                else:
                    filename = file
                filename = filename.replace(".wav", "")
                filename = filename.replace(".mp3", "")
                wav_files.append((file_path, filename))
    return wav_files


def main(args):
    # Initialize the model
    disable_torch_init()

    model = load_pretrained_model(
        args.model_path,
        attn_implementation=args.attn_implementation,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = model.text_decoder.tokenizer

    # model.half()

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

    # Decide which task to perform
    task = args.task.lower()
    if task not in ["tts", "asr", "s2st", "s2st_cloning"]:
        raise ValueError(
            "Invalid task specified. Choose from 'tts', 'asr', 's2st' or 's2st_cloning'."
        )

    if task == "s2st_cloning":
        task = "s2st"
        add_conditioning_audio = True

    # Determine the output directory
    checkpoint_name = "_".join(args.model_path.strip("/").split("/")[-3:])
    force_text_tokens_str = (
        "-force_text_tokens" if args.force_text_tokens else ""
    )
    output_dir = (
        f"{os.getenv('SCRATCH', '.')}/generated"
        f"/{checkpoint_name}-{task}{force_text_tokens_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.condition_audio_path and args.condition_audio_dir:
        logging.warning(
            "Both `--condition-audio-path` and `--condition-audio-dir` are provided. "
            "Ignoring `--condition-audio-path`."
        )

    # Load default condition audio if needed
    speaker_descriptions = None
    condition_audios = None
    if not args.caption_only and task in ["tts"]:
        if not args.condition_audio_dir:
            condition_audio_path = (
                args.condition_audio_path or DEFAULT_AUDIO_PATH
            )
            condition_audios = load_audio_for_conditions(
                condition_audio_path, model.codec_encoder.input_sampling_rate
            )
        else:
            condition_audio_paths = find_audio_files(args.condition_audio_dir)
            condition_audios_samples = []
            for (
                condition_audio_path,
                filename_summary,
            ) in condition_audio_paths:
                condition_audio = load_audio_for_conditions(
                    condition_audio_path,
                    model.codec_encoder.input_sampling_rate,
                )
                condition_audios_samples.append(condition_audio)
    elif task == "tts" and args.speaker_description_path:
        with open(args.speaker_description_path, "r") as f:
            speaker_descriptions = [
                DEFAULT_SPEAKER_DESCRIPTION_PREFIX + line.strip()
                for line in f.readlines()
            ]

    # Prepare the input data
    iterable, data_name = prepare_input_data(args, tokenizer, model, task)

    # Create a subdirectory for the data
    if not os.path.exists(os.path.join(output_dir, data_name)):
        os.makedirs(os.path.join(output_dir, data_name), exist_ok=True)
    output_dir = os.path.join(output_dir, data_name)

    for i, sample in enumerate(tqdm(iterable)):
        if task == "tts":
            if not args.caption_only and args.condition_audio_dir:
                idx = i % len(condition_audios_samples)
                condition_audios = condition_audios_samples[idx]

            if isinstance(sample, dict):
                sentence = sample["transcription"]
                if not args.caption_only and not args.condition_audio_path:
                    print(f"Loading conditioning audio from dataset")
                    condition_audios = load_audio_from_dict(
                        sample,
                        "audio_condition",
                        model.codec_encoder.input_sampling_rate,
                    )

            else:
                sentence = sample

            # Convert numbers to words
            sentence = re.sub(
                r"\d+(\.\d+)?",
                lambda x: num2words(x.group(), lang=args.target_language),
                sentence,
            )
            print(f"Generating output for sample '{sentence}'")
            filename_summary = generate_filename_summary(sentence)
            if args.caption_only:
                spk_dsc = (
                    speaker_descriptions[i % len(speaker_descriptions)]
                    if speaker_descriptions
                    else DEFAULT_SPEAKER_DESCRIPTION
                )
                prompt = f"{TTS_PROMPT}{sentence}\n{spk_dsc}".strip()
            else:
                prompt = f"{TTS_PROMPT}{sentence}\n{CONDITION_PROMPT}".strip()
            kwargs = {
                "tts_input": sentence,
                "force_text_tokens": args.force_text_tokens,
            }
            input_audios = None

        elif task in ["asr"]:
            if isinstance(sample, dict):
                # Sample from dataset
                sentence = sample["transcription"]
                filename_summary = generate_filename_summary(sentence)
                input_audios = load_audio_from_dict(
                    sample,
                    "audio_input",
                    model.audio_encoder.input_sampling_rate,
                )
            else:
                # Sample is a tuple (audio_path, filename)
                audio_path, filename_summary = sample
                input_audios = load_audio_for_asr(
                    audio_path, model.audio_encoder.input_sampling_rate
                )
            prompt = ASR_PROMPT.strip()
            kwargs = {}

        elif task == "s2st":
            if isinstance(sample, dict):
                # Sample from dataset
                text_input = sample["text_input"]
                text_output = sample["text_output"]
                source_language = sample["source_language"]
                target_language = sample["target_language"]
                filename_summary = (
                    generate_filename_summary(text_input)
                    + "_"
                    + source_language
                    + "_"
                    + target_language
                )
                input_audios = load_audio_from_dict(
                    sample,
                    "audio_input",
                    model.audio_encoder.input_sampling_rate,
                )
                # use input_audios to condition the model
                if add_conditioning_audio:
                    condition_audios = load_audio_for_conditions(
                        audio_path, model.codec_encoder.input_sampling_rate
                    )
            else:
                # Sample is a tuple (audio_path, filename)
                text_input = None
                text_output = None
                audio_path, filename_summary = sample
                input_audios = load_audio_for_asr(
                    audio_path, model.audio_encoder.input_sampling_rate
                )
                # use input_audios to condition the model
                if add_conditioning_audio:
                    condition_audios = load_audio_for_conditions(
                        audio_path, model.codec_encoder.input_sampling_rate
                    )
                source_language = args.source_language
                target_language = args.target_language

            # Map language codes to language names if necessary
            source_language_name = LANGUAGES_CODE_NAME.get("en", {}).get(
                source_language, source_language
            )
            target_language_name = LANGUAGES_CODE_NAME.get("en", {}).get(
                target_language, target_language
            )
            s2st_prompt = (
                S2ST_PROMPT_TEMPLATE.format(
                    source_language=source_language_name,
                    target_language=target_language_name,
                )
                if not add_conditioning_audio
                else S2ST_CLONING_PROMPT_TEMPLATE.format(
                    source_language=source_language_name,
                    target_language=target_language_name,
                )
            )
            prompt = f"{s2st_prompt}".strip()
            kwargs = {
                "tts_input": "",
                "force_text_tokens": False,
            }  # Empty tts_input for S2ST

        else:
            continue  # Skip unknown tasks

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        prompt_text = conv.get_prompt()
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            model.device
        )
        streamer = None

        with torch.inference_mode():
            print(
                f"Processing '{task.upper()}' task for input: '{filename_summary}'"
            )
            output = model.generate(
                input_ids,
                images=None,
                image_sizes=None,
                audios=input_audios if task in ["asr", "s2st"] else None,
                condition_audios=(
                    condition_audios if task in ["tts", "s2st"] else None
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                tts_max_len=args.tts_max_len,
                streamer=streamer,
                use_cache=True,
                **kwargs,
            )
            # Save conditioning audio or speaker description if applicable
            if task in ["tts", "s2st"]:
                if not args.caption_only:
                    output_path = os.path.join(
                        output_dir, f"{filename_summary}.wav"
                    )
                    output_path = increment_path(output_path)
                    output_path = output_path.replace(
                        ".wav", "_conditioning.wav"
                    )
                    if condition_audios:
                        torchaudio.save(
                            output_path,
                            condition_audios[0][0]
                            .to(dtype=torch.float32)
                            .detach()
                            .cpu(),
                            condition_audios[0][1],
                        )
                        print("Voice conditioning saved at: ", output_path)
                    else:
                        print("No voice conditioning audio was loaded.")
                else:
                    output_path = os.path.join(
                        output_dir,
                        f"{filename_summary}_speaker_description.txt",
                    )
                    output_path = increment_path(output_path)
                    with open(output_path, "w") as f:
                        f.write(DEFAULT_SPEAKER_DESCRIPTION)
                    print("Speaker description saved at: ", output_path)

            # Save the generated audio
            if task in ["tts", "s2st"]:
                audio_values, output_ids = output
                if audio_values is not None:
                    for audio_value in audio_values:
                        output_path = os.path.join(
                            output_dir, f"{filename_summary}.wav"
                        )
                        output_path = increment_path(output_path)
                        torchaudio.save(
                            output_path,
                            audio_value.to(dtype=torch.float32).detach().cpu(),
                            24000,
                        )
                        print(f"Audio saved at: {output_path}")
                        try:
                            output_text = tokenizer.decode(output_ids).strip()
                        except Exception as e:
                            print("Error decoding output: ", e)
                            output_text = None
                        # Save the text in a txt file with the same name
                        with open(
                            output_path.replace(".wav", ".txt"), "w"
                        ) as f:
                            if task == "tts":
                                f.write(sentence)
                                # if output_text:
                                #     f.write(f"\nOutput:{output_text}")
                            elif task == "s2st":
                                f.write(
                                    f"Source Language: {source_language}\n"
                                )
                                f.write(
                                    f"Target Language: {target_language}\n"
                                )
                                if text_input:
                                    f.write(f"Text Input:{text_input}\n")
                                if text_output:
                                    f.write(f"GT Text Output:{text_output}\n")
                                if output_text:
                                    f.write(f"Output:{output_text}\n")
                        print(
                            "\n",
                            {"prompt": prompt, "outputs": output_text},
                            "\n",
                        )
            elif task == "asr":
                output_ids = output[0]
                try:
                    output_text = tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip()
                except Exception as e:
                    print("Error decoding output: ", e)
                    output_text = None
                # Save the transcribed text
                output_path = os.path.join(
                    output_dir, f"{filename_summary}.txt"
                )
                output_path = increment_path(output_path)
                with open(output_path, "w") as f:
                    f.write(output_text)
                print(f"Transcription saved at: {output_path}")
                print(
                    "\n",
                    {"prompt": prompt, "transcription": output_text},
                    "\n",
                )

    print("All outputs have been generated and saved.")


def prepare_input_data(args, tokenizer, model, task):
    """
    Prepare the input data based on provided arguments.
    """
    if args.dataset_config is not None and (
        args.sentences_file_path is not None
        or args.audios_file_path is not None
    ):
        logging.warning(
            "Both `--dataset-config` and `--sentences-file-path` or `--audios-file-path` are provided. "
            "Ignoring `--sentences-file-path` and `--audios-file-path`."
        )

    # Sentences or audio files to process
    iterable = None
    if args.dataset_config is not None:
        dataset_config = args.dataset_config
        dataloader = get_inference_dataloader(dataset_config, model)
        # get an iterator from the dataloader
        iterable = iter(dataloader)
        data_name = os.path.basename(args.dataset_config).replace(".yml", "")
    elif task == "tts":
        # Handle TTS inputs
        if args.sentences_file_path:
            with open(args.sentences_file_path, "r") as f:
                sentences = f.readlines()
                sentences = [sentence.strip() for sentence in sentences]
            iterable = sentences
            data_name = os.path.basename(args.sentences_file_path).replace(
                ".txt", ""
            )
        else:
            # Default sentences for TTS
            sentences = [
                "Most of the time travelers worry about their luggage.",
                "Hello.",
                "Thank you.",
                "Hello! This is customer support. How can I help you today?",
                "What is the meaning of life if you are not happy?",
                "The most important thing is to enjoy your life. To be happy. It's all that matters.",
            ]
            iterable = sentences
            data_name = "default"
    elif task in ["asr", "s2st"]:
        # Handle ASR and S2ST inputs
        if args.audios_file_path:
            # args.audios_file_path is a directory; find all .wav files recursively
            audio_files = find_audio_files(args.audios_file_path)
            if not audio_files:
                raise ValueError(
                    f"No .wav files found in directory {args.audios_file_path}"
                )
            iterable = audio_files
            data_name = os.path.basename(args.audios_file_path)
        elif args.sentences_file_path:
            with open(args.sentences_file_path, "r") as f:
                audio_paths = f.readlines()
                audio_paths = [path.strip() for path in audio_paths]
            # For consistency, construct filename summaries
            audio_files = []
            for audio_path in audio_paths:
                filename = os.path.basename(audio_path)
                audio_files.append((audio_path, filename))
            iterable = audio_files
            data_name = os.path.basename(args.sentences_file_path).replace(
                ".txt", ""
            )
        else:
            raise ValueError(
                "For ASR and S2ST tasks, please provide either --audios-file-path or --sentences-file-path pointing to audio files."
            )
    else:
        iterable = []
        data_name = "default"

    return iterable, data_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to perform: 'tts', 'asr', 's2st' or 's2st_cloning'",
    )
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--caption-only", action="store_true")
    parser.add_argument("--tts-max-len", type=int, default=60)
    parser.add_argument("--force-text-tokens", action="store_true")
    parser.add_argument(
        "--sentences-file-path",
        type=str,
        default=None,
        help="Path to a file containing sentences or audio paths",
    )
    parser.add_argument(
        "--speaker-description-path",
        type=str,
        default=None,
        help="Path to a file containing speaker description",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to dataset configuration for evaluation",
    )
    parser.add_argument(
        "--condition_audio_path",
        type=str,
        default=None,
        help="Path to condition audio for voice cloning",
    )
    parser.add_argument(
        "--condition_audio_dir",
        type=str,
        default=None,
        help="Directory containing condition audio for voice cloning",
    )
    parser.add_argument(
        "--audios-file-path",
        type=str,
        default=None,
        help="Directory containing audio files for ASR and S2ST tasks",
    )
    # Additional arguments for S2ST
    parser.add_argument(
        "--source-language",
        type=str,
        default=None,
        help="Source language code for S2ST",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default=None,
        help="Target language code for S2ST or TTS",
    )
    args = parser.parse_args()

    if args.task == "s2st":
        if args.source_language not in SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(
                f"Source language '{args.source_language}' is not supported. "
                f"Please choose from {SUPPORTED_SOURCE_LANGUAGES}."
            )
        if args.target_language != SUPPORTED_TARGET_LANGUAGE:
            raise ValueError(
                f"Target language '{args.target_language}' is not supported. "
                f"Please choose '{SUPPORTED_TARGET_LANGUAGE}'."
            )
    if args.task == "tts":
        if args.target_language not in SUPPORTED_TTS_LANGUAGES:
            raise ValueError(
                f"Target language '{args.target_language}' is not supported. "
                f"Please choose '{SUPPORTED_TTS_LANGUAGES}'."
            )

    main(args)


# python $SPEECHLMM_ROOT/scripts/inference.py \
#   --model-path "$CHECKPOINTS_HOME/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_speaker_description/checkpoint-30000" \
#   --task "tts" \
#   --conv-mode "llama_3_1" \
#   --tts-max-len 20 \
#   --caption-only \
#   --sentences-file-path "$SCRATCH/en_tts_spk_dsc.txt" \
#   --target-language "en"


# write a script to do inference on LibriTTS_R dataset
# python $SPEECHLMM_ROOT/scripts/inference.py \
#   --model-path "$CHECKPOINTS_HOME/moshi/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io/moshi_bert_speaker_description/checkpoint-30000" \
#   --task "tts" \
#   --conv-mode "llama_3_1" \
#   --tts-max-len 30 \
#   --caption-only \
#   --dataset_config "$SPEECHLMM_ROOT/conf/datasets/libritts_r_test.yml" \
#   --target-language "en"

# python $SPEECHLMM_ROOT/scripts/inference.py \
#   --model-path "$CHECKPOINTS_HOME/speech2speech/speechlmm-pretrain-audio-seamless-mlp-llama_3_1-moshi_bert-qformer-features-speech2speech/moshi_bert_s2st_prefix_libritts_same_io/checkpoint-33000" \
#   --task "s2st_cloning" \
#   --conv-mode "llama_3_1" \
#   --tts-max-len 30 \
#   --dataset_config "$SPEECHLMM_ROOT/conf/datasets/cvss_test.yml" \
