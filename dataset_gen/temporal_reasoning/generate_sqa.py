import argparse
import ast
import json
import os
import random
import time
from typing import Dict, List

import numpy
import pandas
import torch
import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_prompts(prompts_dir):
    prompts = {}
    prompts["system_message"] = (
        open(os.path.join(prompts_dir, "system_message.txt")).read().strip()
    )
    prompts["few_shots"] = {}
    for few_shots in os.listdir(os.path.join(prompts_dir)):
        if few_shots.endswith("caps.txt"):
            few_shots_name = few_shots.split("_")[0]
            prompts["few_shots"][few_shots_name] = {}
            prompts["few_shots"][few_shots_name]["caps"] = (
                open(os.path.join(prompts_dir, few_shots)).read().strip()
            )
            prompts["few_shots"][few_shots_name]["conv"] = (
                open(
                    os.path.join(
                        prompts_dir,
                        few_shots.replace("caps", "conv"),
                    )
                )
                .read()
                .strip()
            )
    return prompts


def load_data(data_dir, input_file):
    data_path = os.path.join(data_dir, input_file)
    data = json.load(open(data_path))
    return data


def build_prompt(prompts, timestamps, tokenizer):
    if model_name != "meta-llama/Meta-Llama-3-8B-Instruct":
        raise ValueError("Only Meta-Llama-3-8B-Instruct is supported")

    system_message = prompts["system_message"]
    messages = [
        {"role": "system", "content": system_message},
    ]

    few_shots = prompts["few_shots"]

    for few_shots_name in few_shots:
        caps = few_shots[few_shots_name]["caps"]
        conv = few_shots[few_shots_name]["conv"]

        messages.append({"role": "user", "content": caps})
        messages.append({"role": "assistant", "content": conv})

    # timestamps text format:
    # 1. text: "Hello, everyone!", start: 0.0, end: 1.5
    # 2. text: "Welcome to the conference.", start: 1.5, end: 3.5
    # timestamps_text = "\n".join(
    #     [
    #         f"{i+1}. text: \"{t['text']}\", start: {t['start']}, end: {t['end']}"
    #         for i, t in enumerate(timestamps)
    #     ]
    # )

    # apply ast.literal_eval to convert string (t) to list
    # print(timestamps)
    timestamps_text = "\n".join(
        [
            f"{i+1}. text: \"{t['text']}\", start: {t['start']}, end: {t['end']}"
            for i, t in enumerate(ast.literal_eval(timestamps))
        ]
    )

    messages.append({"role": "user", "content": timestamps_text})

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_name,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # print(prompt)
    # exit(0)
    return prompt


def generate_prompts(prompts, data):
    jsons = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for video in data:
        for audio in data[video]:
            timestamps = audio["timestamps"]

            prompt = build_prompt(prompts, timestamps, tokenizer)
            jsons.append(
                {
                    "id": audio["id"],
                    "audio": audio["audio"],
                    "timestamps": timestamps,
                    "prompt": prompt,
                }
            )

    out_file_name = f"temporal_reasoning_qa_prompts.jsonl"
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

        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],  # KEYPOINT HERE
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
        print(f"Processing {len(outputs['id'])} outputs")
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
        default="meta-llama/Meta-Llama-3-8B-Instruct",
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
        prompts = load_prompts(prompts_dir)
        data = load_data(data_dir, input_file)
        generate_prompts(prompts, data)

    if args.generate_sqa:
        if os.path.exists(
            os.path.join(output_dir, "sqa_generated_output.jsonl")
        ):
            os.remove(os.path.join(output_dir, "sqa_generated_output.jsonl"))

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


# python ./dataset/temporal_reasoning/generate_sqa.py --generate_prompts --data_dir $DATA_HOME/temporal_reasoning/ --input_file generation_ready_data_2.json --prompts_dir ./dataset/temporal_reasoning/prompts --output_dir $DATA_HOME/temporal_reasoning/


#  python ./dataset/temporal_reasoning/generate_sqa.py --generate_sqa \
#       --input_file $DATA_HOME/temporal_reasoning/temporal_reasoning_qa_prompts.jsonl \
#       --model_name meta-llama/Meta-Llama-3-8B-Instruct \
#       --n_gpus 1 \
#       --output_dir $DATA_HOME/temporal_reasoning/ \
#       --temperature 0.5 \
#   --top_k <topk> \
#   --top_p <topp> \
