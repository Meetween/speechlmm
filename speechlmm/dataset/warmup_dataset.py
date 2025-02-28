# This script is used only to warm up the dataset, it is not used in the training process
# It should be launched in a separate job on the cluster (even with 0 GPUs) before the training job
import argparse
import os

from speechlmm.arguments import DataArguments
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--rebuild_dataset_cache", action="store_true", default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    config_path = args.config_path
    # if an env variable is in the config path, replace it with the value
    config_path = os.path.expandvars(config_path)

    data_args = DataArguments(
        sampling_rate=16000,
        data_config_path=config_path,
        is_multimodal=True,
        dataloader_debug=True,
        organize_eval_dataset_per_task=True,
        rebuild_dataset_cache=args.rebuild_dataset_cache,
    )
    print(f"{data_args}")
    datasets = DatasetsWrapper(data_args)
