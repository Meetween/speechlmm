import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from speechlmm.eval.evaluator import Evaluator
from speechlmm.eval.metrics import (
    EvalOnGeneratePrediction,
    compute_asr_metrics,
    compute_st_metrics,
)


def read_json_as_dict(filepath):
    with open(filepath, "r", encoding="utf-8") as json_file:
        preds = json.load(json_file)
    dict = defaultdict(list)

    for pred in preds:
        dict["prompts"].append(pred["prompt"])
        dict["source_langs"].append(pred["source_language"])
        dict["target_langs"].append(pred["target_language"])
        dict["gts"].append(pred["groundtruth"])
        dict["preds"].append(pred["predictions"])
        dict["tasks"].append(pred["task"])
        dict["transcriptions"].append(pred["transcription"])

    return dict


def compute_metrics(predictions_path, out=None, compute_comet=True):
    dict = read_json_as_dict(predictions_path)
    all_metrics = Evaluator.compute_metrics(dict, compute_comet)

    if out is None:
        # Save the results in the same directory as the predictions file
        predictions_dir, predictions_file = os.path.split(predictions_path)

        if predictions_file.endswith("preds.json"):
            output_path = predictions_file.replace("preds.json", "scores.json")
        else:
            output_path = predictions_file.replace(".json", "_scores.json")
        output_path = os.path.join(predictions_dir, output_path)
    print(f"Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(all_metrics, json_file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds", type=str, help="Path to the predictions file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output file", required=False
    )
    parser.add_argument(
        "--no-comet", action="store_true", help="Do not compute COMET metrics"
    )
    args = parser.parse_args()
    predictions_path = args.preds
    output_path = args.output
    compute_comet = not args.no_comet

    compute_metrics(predictions_path, output_path, compute_comet)


if __name__ == "__main__":
    main()
