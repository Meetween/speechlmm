# This script is used only to warm up the dataset, it is not used in the training process
# It should be launched in a separate job on the cluster (even with 0 GPUs) before the training job
import argparse
from pathlib import Path

from speechlmm.arguments import DataArguments
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper


def optional(type_):
    def parse(value_str):
        if value_str.lower() == "none":
            return None
        return type_(value_str)

    return parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="path_to_config", type=Path, required=True
    )
    parser.add_argument(
        "--audio-input-sampling-rate", type=int, required=False
    )
    parser.add_argument("--codec-sampling-rate", type=int, required=False)
    parser.add_argument("--codec-frame-rate", type=float, required=False)
    parser.add_argument(
        "--cache-final-datasets", action="store_true", default=True
    )
    parser.add_argument("--num-proc", type=optional(int), default=None)
    parser.add_argument("--rebuild-cache", action="store_true", default=False)
    args = parser.parse_args()

    data_args = DataArguments(
        audio_input_sampling_rate=args.audio_input_sampling_rate,
        codec_frame_rate=args.codec_frame_rate,
        codec_sampling_rate=args.codec_sampling_rate,
        data_config_path=args.path_to_config,
        is_multimodal=True,
        # group_dataset_by_task={
        #     "train": args.group_dataset_by_task_train,
        #     "eval": args.group_dataset_by_task_eval,
        #     "test": args.group_dataset_by_task_test,
        # },
        cache_final_datasets=args.cache_final_datasets,
        num_proc_for_preprocessing=args.num_proc,
        rebuild_dataset_cache=args.rebuild_cache,
    )
    datasets = DatasetsWrapper(data_args, load_only=True)
