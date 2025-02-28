import json
import os

import wandb


def log_metrics(results_dir, wandb_project, wandb_run_name, wandb_run_id):
    model_name = results_dir.split("/")[-1]
    # get all scores files in the dir
    json_files = [
        f for f in os.listdir(results_dir) if f.endswith("_scores.json")
    ]
    table = wandb.Table(
        columns=[
            "model",
            "dataset_name",
            "task",
            "source_language",
            "target_language",
            "wer",
            "bleu",
            "comet",
        ]
    )
    for json_file in json_files:
        with open(os.path.join(results_dir, json_file), "r") as f:
            results = json.load(f)

        for result in results:
            wer = result["metrics"].get("wer", None)
            bleu = result["metrics"].get("bleu", None)
            comet = result["metrics"].get("comet", None)
            dataset_name = json_file.split("_scores.json")[0]
            row = [
                model_name,
                dataset_name,
                result["task"],
                result["source_language"],
                result["target_language"],
                wer,
                bleu,
                comet,
            ]
            table.add_data(*row)

    if wandb_run_id:
        wandb.init(project=wandb_project, id=wandb_run_id, resume=True)
    elif wandb_run_name:
        wandb.init(project=wandb_project, name=wandb_run_name)
    else:
        raise ValueError(
            "You need to provide a wandb_run_id or a wandb_run_name"
        )
    wandb.log({"scores": table})
    wandb.finish()


# main add argparse
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument(
        "--wandb_run_name", type=str, required=False, default=None
    )
    parser.add_argument(
        "--wandb_run_id", type=str, required=False, default=None
    )
    args = parser.parse_args()
    log_metrics(
        args.results_dir,
        args.wandb_project,
        args.wandb_run_name,
        args.wandb_run_id,
    )


if __name__ == "__main__":
    main()
