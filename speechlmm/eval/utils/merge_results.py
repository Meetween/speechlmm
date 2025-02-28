import argparse
import json
import os
from typing import Dict


def merge_results(
    dir_path, remove_scores_files=False, merge_preds=False, ds_name=None
):
    files = os.listdir(dir_path)
    datasets = set()

    preds_files = []
    scores_files = []
    for file in files:
        if file.endswith("_preds.json"):
            dataset_name = file.split("_")[0]
            if ds_name != dataset_name:
                continue
            if file.endswith("_percent_preds.json"):
                percent = file.split("_")[-3]
                dataset_name = dataset_name + "_" + percent + "_percent"

            datasets.add(dataset_name)
            preds_files.append(file)
        if file.endswith("_scores.json"):
            dataset_name = file.split("_")[0]
            if ds_name != dataset_name:
                continue
            if file.endswith("_percent_scores.json"):
                percent = file.split("_")[-3]
                dataset_name = dataset_name + "_" + percent + "_percent"
            datasets.add(dataset_name)
            scores_files.append(file)

    for dataset_name in datasets:
        if merge_preds:
            preds = []
            for preds_file in preds_files:
                if not preds_file.startswith(dataset_name.split("_")[0]):
                    continue
                with open(os.path.join(dir_path, preds_file), "r") as f:
                    try:
                        json_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading {preds_file}")
                    assert isinstance(
                        json_data, list
                    ), "preds json data should be a list"
                    preds.extend(json_data)

            with open(
                os.path.join(
                    dir_path,
                    dataset_name + "_preds.json",
                ),
                "w",
            ) as f:
                json.dump(preds, f, indent=4, ensure_ascii=False)

        # merge scores files
        scores = []
        for scores_file in scores_files:
            if not scores_file.startswith(dataset_name.split("_")[0]):
                continue
            with open(os.path.join(dir_path, scores_file), "r") as f:
                try:
                    json_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {scores_file}")
                assert isinstance(
                    json_data, list
                ), "scores json data should be a list"
                scores.extend(json_data)
        with open(
            os.path.join(
                dir_path,
                dataset_name + "_scores.json",
            ),
            "w",
        ) as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)

        if remove_scores_files:
            for scores_file in scores_files:
                os.remove(os.path.join(dir_path, scores_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--merge-preds", action="store_true", default=False)
    parser.add_argument(
        "--remove-scores-files",
        action="store_true",
        help="Remove scores files after merging",
        default=False,
        required=False,
    )
    # specify a dataset name to merge
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Specify a dataset name to merge",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    remove_scores_files = args.remove_scores_files
    merge_preds = args.merge_preds
    merge_results(
        results_dir, remove_scores_files, merge_preds, args.dataset_name
    )


if __name__ == "__main__":
    main()
