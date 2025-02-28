import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy
import pandas
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_prompts(prompts_dir):
    prompts = {}
    task_weights = json.load(
        open(os.path.join(prompts_dir, "task_weights.json"))
    )
    tasks = list(task_weights.keys())
    weights = list(task_weights.values())

    for task in tasks:
        prompts[task] = {}
        prompts[task]["system_message"] = (
            open(os.path.join(prompts_dir, task, "system_message.txt"))
            .read()
            .strip()
        )
        prompts[task]["few_shots"] = {}
        for few_shots in os.listdir(os.path.join(prompts_dir, task)):
            if few_shots.endswith("caps.txt"):
                few_shots_name = few_shots.split("_")[0]
                prompts[task]["few_shots"][few_shots_name] = {}
                prompts[task]["few_shots"][few_shots_name]["caps"] = (
                    open(os.path.join(prompts_dir, task, few_shots))
                    .read()
                    .strip()
                )
                prompts[task]["few_shots"][few_shots_name]["conv"] = (
                    open(
                        os.path.join(
                            prompts_dir,
                            task,
                            few_shots.replace("caps", "conv"),
                        )
                    )
                    .read()
                    .strip()
                )
    return tasks, weights, prompts


def load_data(data_dir):
    data_path = os.path.join(data_dir, "libritts_sqa.json")
    data = json.load(open(data_path))
    return data


def build_prompt(prompts, transcription, task, tokenizer):
    system_message = prompts[task]["system_message"]
    few_shots = prompts[task]["few_shots"]
    examples = "###" + "\n" if "mistral" in model_name else ""
    for i, few_shots_name in enumerate(few_shots):
        few_shots_caps = few_shots[few_shots_name]["caps"]
        few_shots_conv = few_shots[few_shots_name]["conv"]
        examples += "\n# Example " + str(i + 1) + "\n"
        examples += "\n# Speech transcription\n" + few_shots_caps + "\n"
        if task == "detail_description":
            examples += "\n# Detailed description\n" + few_shots_conv + "\n"
        else:
            examples += "\n" + few_shots_conv + "\n"
    examples += "###" if "mistral" in model_name else ""

    instruction = "\n# Speech transcription\n" + transcription + "\n"

    if "mistral" in model_name:
        chat = [
            {
                "role": "user",
                "content": system_message
                + "\n"
                + examples
                + "\n"
                + instruction,
            },
        ]
        chat_template = tokenizer.apply_chat_template(chat, tokenize=False)
        return chat_template
    else:
        chat = [
            {"role": "system", "content": system_message + "\n" + examples},
            {"role": "user", "content": instruction},
        ]
        chat_template = tokenizer.apply_chat_template(chat, tokenize=False)
        return chat_template


def generate_prompts(tasks, weights, prompts, data):
    jsons = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for audio in data:
        transcription = audio["transcription"]

        task = random.choices(tasks, weights=weights, k=1)[0]
        # choose a random few shot
        # few_shot = random.choice(list(few_shots.keys()))
        # example = "Speech:" + few_shots[few_shot]["caps"] + "\n" + few_shots[few_shot]["conv"] + "\n"

        prompt = build_prompt(prompts, transcription, task, tokenizer)
        jsons.append(
            {
                "id": audio["id"],
                "audio": audio["audio"],
                "transcription": transcription,
                "task": task,
                "prompt": prompt,
            }
        )

    jsons = sorted(
        jsons, key=lambda x: x["task"]
    )  # to optimize inference (prefix will be cached)
    out_file_name = f"sqa_prompts_{model_name}.jsonl"
    with open(os.path.join(output_dir, out_file_name), "w") as f:
        for obj in jsons:
            json.dump(obj, f)
            f.write("\n")  # Add a newline for separation


class LLMPredictor:
    def __init__(
        self,
        model_name,
        tensor_parallel_size,
        seed,
        temperature,
        top_p,
        top_k,
        max_tokens,
        dtype=torch.float16,
    ):
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            device="cuda",
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )

    def __call__(self, batch: Dict[str, numpy.ndarray]) -> Dict[str, List]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["prompt"], self.sampling_params)

        generated_text = []
        for i, output in enumerate(outputs):
            generated_text.append(" ".join([o.text for o in output.outputs]))

        return {
            "id": batch["id"],
            "audio": batch["audio"],
            "generated_text": generated_text,
        }


def process_output(outputs):
    with open(
        os.path.join(output_dir, "sqa_generated_output.jsonl"), "a"
    ) as f:
        ids, audios, generated_texts = (
            outputs["id"],
            outputs["audio"],
            outputs["generated_text"],
        )
        for i in range(len(ids)):
            obj = {
                "id": ids[i],
                "audio": audios[i],
                "generated_text": generated_texts[i],
            }
            json.dump(obj, f)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_prompts", action="store_true", help="Generate prompts"
    )
    parser.add_argument(
        "--generate_sqa", action="store_true", help="Generate outputs"
    )

    parser.add_argument(
        "--data_dir", type=str, help="Path to the data directory"
    )
    parser.add_argument(
        "--prompts_dir", type=str, help="Path to the prompts directory"
    )

    parser.add_argument(
        "--input_file", type=str, help="Path to the input file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", default=256
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature", default=0.3
    )
    parser.add_argument("--top_p", type=float, help="Top p", default=0.95)
    parser.add_argument("--top_k", type=int, help="Top k", default=-1)
    parser.add_argument(
        "--max_gen_tokens", type=int, help="Max generated tokens", default=1000
    )
    parser.add_argument("--n_gpus", type=int, help="Number of GPUs", default=1)
    parser.add_argument(
        "--output_dir", type=str, help="Path to the output directory"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    prompts_dir = args.prompts_dir
    output_dir = args.output_dir
    input_file = args.input_file

    model_name = args.model_name

    batch_size = args.batch_size
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_gen_tokens = args.max_gen_tokens
    n_gpus = args.n_gpus

    seed = 42

    if args.generate_prompts:
        tasks, weights, prompts = load_prompts(prompts_dir)
        data = load_data(data_dir)
        generate_prompts(tasks, weights, prompts, data)

    if args.generate_sqa:
        if os.path.exists(
            os.path.join(output_dir, "sqa_generated_output.jsonl")
        ):
            os.remove(os.path.join(output_dir, "sqa_generated_output.jsonl"))

        ray.init(_temp_dir=f"{os.getenv('SCRATCH')}/ray_temp")

        llm = LLMPredictor(
            model_name, n_gpus, seed, temperature, top_p, top_k, max_gen_tokens
        )
        ds = pandas.read_json(input_file, lines=True, dtype=str)
        ds = ds.to_dict(orient="records")

        start = time.time()

        for i in range(0, len(ds), batch_size):
            batch = ds[i : i + batch_size]
            batch = {
                "id": [b["id"] for b in batch],
                "audio": [b["audio"] for b in batch],
                "prompt": [b["prompt"] for b in batch],
            }
            process_output(llm(batch))

        end = time.time()

        print(f"Time taken: {end - start}")
