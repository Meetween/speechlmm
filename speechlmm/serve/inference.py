import argparse
import json
import os

import torch
from tqdm import tqdm

from speechlmm.model.builder import load_pretrained_model
from speechlmm.serve.custom_audio_dataset import AudioDataset
from speechlmm.serve.prompt_utils import PromptBuilder
from speechlmm.utils import disable_torch_init, torch_dtype_from_str

# Load the inputs text list from JSON file
INPUTS_TEXT_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "INPUTS_TEXT_LIST.json"
)
with open(INPUTS_TEXT_LIST_PATH, "r") as f:
    INPUTS_TEXT_LIST = json.load(f)

ALLOWED_TASKS = ["ASR", "ST", "SLU", "SSUM", "VSR", "SQA"]


def handle_output_path(args):
    scratch = os.environ["SCRATCH"]
    model_name = args.model_path.replace(f"{scratch}/", "").replace("/", "_")
    output_dir = os.path.join(args.output_path, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    filename = f"{args.task}_{args.language}"
    if args.task == "ST":
        filename += f"_{args.source_language}_{args.target_language}"
    filename += ".json"
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        response = input(
            f"\n ** WARNING: Output file {output_path} already exists. \nDo you want to overwrite it? If not, results will be appended to the file. \nDefault y (y/n): "
        )
        if response == "y" or response == "":
            os.remove(output_path)
    return output_path


def main(args):
    output_path = handle_output_path(args)
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
    # handle tokenizer pad side
    model.text_decoder.tokenizer.padding_side = "left"
    model.config.tokenizer_padding_side = "left"

    conv_mode = model.text_decoder.conversation_version

    user_TASK = args.task
    language = args.language
    source_language = args.source_language
    target_language = args.target_language
    prompt_builder = PromptBuilder(
        task=user_TASK,
        conv_mode=conv_mode,
        language=language,
        source_language=source_language,
        target_language=target_language,
        tokenizer=model.text_decoder.tokenizer,
    )
    audio_dataset = AudioDataset(
        audio_path=args.input_path,
        prompt_builder=prompt_builder,
        target_sr=model.audio_encoder.input_sampling_rate,
    )
    audio_dataloader = audio_dataset.get_data_loader(
        batch_size=args.batch_size
    )
    for batch in tqdm(audio_dataloader):
        with torch.inference_mode():
            output_ids = model.generate(
                batch["input_ids"],
                audios=batch["audios_srs"],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

        outputs = model.text_decoder.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        results = []
        for audio_path, task, output in zip(
            batch["audio_path"], batch["task"], outputs
        ):
            results.append(
                {
                    "audio_path": audio_path,
                    "task": task,
                    "output": output,
                }
            )
            # if args.output_sort == "task_first":
            #     if task not in results:
            #         results[task] = {}
            #     results[task][audio_path] = output
            # else:  # audio_first
            #     if audio_path not in results:
            #         results[audio_path] = {}
            #     results[audio_path][task] = output

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_results = json.load(f)
            results.extend(existing_results)
        with open(output_path, "w") as f:
            json.dump(results, indent=2, fp=f)
    print(f"Results saved to {output_path}")


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--output-sort",
        type=str,
        default="audio_first",
        choices=["task_first", "audio_first"],
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="debug mode"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=ALLOWED_TASKS + [""],
        default="",
        help=f"task: {ALLOWED_TASKS}",
    )
    parser.add_argument(
        "-l", "--language", type=str, default="en", help="language"
    )
    parser.add_argument(
        "-sl",
        "--source-language",
        type=str,
        default="",
        help="source language",
    )
    parser.add_argument(
        "-tl",
        "--target-language",
        type=str,
        default="",
        help="target language",
    )
    parser.add_argument(
        "-p",
        "--input-path",
        type=str,
        default="",
        help="path to the audio file or folder of audio files",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="",
        help="path to the output json file",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )
    args = parser.parse_args()
    main(args)

# sample usage:
# ASR
# python speechlmm/serve/inference.py --model-path "$SCRATCH/checkpoints/speechlmm-v1/checkpoints/v1/l/checkpoint-1500" \
# --input-path "$SCRATCH/test_audios" -o "$SCRATCH/eval/speechlmm" --task ASR --language en

# ST
# python speechlmm/serve/inference.py --model-path "$SCRATCH/checkpoints/speechlmm-v1/checkpoints/v1/l/checkpoint-1500" \
# --input-path "$SCRATCH/test_audios" -o "$SCRATCH/eval/speechlmm" --task ST --language en --source-language en --target-language es
