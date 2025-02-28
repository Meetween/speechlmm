import os
from collections import defaultdict

from datasets import Audio, Video, concatenate_datasets

from speechlmm.arguments import DataArguments
from speechlmm.dataset.config import CustomDatasetConfig, DatasetsWrapperConfig
from speechlmm.dataset.dataset_mapping import DATASET_MAPPING


class DatasetsWrapper:
    def __init__(self, data_args: DataArguments, load_only=False):
        self.data_args = data_args

        self.config = DatasetsWrapperConfig(
            self.data_args.data_config_path
        ).from_yml()

        self.train_dataset = None
        self.test_dataset = None
        self.eval_dataset = None

        # Load individual datasets, also caching them if needed
        datasets = []
        for DATA in self.config.DATA:
            for dataset_name, dataset_config in DATA.items():
                dataset_config["OUTPUTS_TEXT_LIST"] = (
                    self.config.OUTPUTS_TEXT_LIST
                )
                dataset_config["INPUTS_TEXT_LIST"] = (
                    self.config.INPUTS_TEXT_LIST
                )

                dataset_cls = DATASET_MAPPING.get(dataset_name, None)
                if dataset_cls is None:
                    raise ValueError(
                        f"Dataset '{dataset_name}' does not exist. Valid "
                        f"datasets are {list(DATASET_MAPPING.keys())}"
                    )

                assert all(
                    p["destination"] in dataset_cls.splits
                    for p in dataset_config["partitions"].values()
                ), (
                    f"Invalid destination in partitions. Valid destinations "
                    f"are {dataset_cls.splits}"
                )

                additional_args = dataset_config.pop("additional_args", {})
                additional_args.update(
                    {
                        "sampling_rate": self.data_args.audio_input_sampling_rate,
                    }
                )
                datasets.append(
                    dataset_cls(
                        CustomDatasetConfig(
                            **dataset_config,
                            additional_args=additional_args,
                        ),
                        cache_final_datasets=self.data_args.cache_final_datasets,
                        rebuild_cache=self.data_args.rebuild_dataset_cache,
                        num_proc_for_preprocessing=self.data_args.num_proc_for_preprocessing,
                    )
                )

        if load_only:
            return

        # Concatenate them
        for split in ["train", "test", "eval"]:
            dataset = None

            datasets_by_task = defaultdict(list)
            group_by_task = self.data_args.group_dataset_by_task[split]
            for ds in datasets:
                ds_split = getattr(ds, f"{split}_dataset")
                if ds_split is not None:
                    task = ds.config.task if group_by_task else "all_tasks"
                    datasets_by_task[task].append(ds_split)
            setattr(
                self,
                f"{split}_dataset",
                {
                    task: self.concat_datasets(split, task_datasets)
                    for task, task_datasets in datasets_by_task.items()
                },
            )

        if self.data_args.dataloader_debug:

            def log_dataset(name, dataset):
                if dataset:
                    print(f"{name} dataset: {len(dataset)} samples")

            def log_samples(name, dataset, n_samples=5):
                if dataset:
                    sample = dataset.shuffle(seed=42).select(range(n_samples))
                    for i, sample_data in enumerate(sample):
                        print(f"{name} dataset sample {i}: {sample_data}")

            if data_args.dataloader_debug:
                for dataset in datasets:
                    print(
                        f"----- Dataset: {dataset.__class__.__name__} Task: {dataset.config.task} -----"
                    )
                    log_dataset("Train", dataset.train_dataset)
                    log_dataset("Test", dataset.test_dataset)
                    if isinstance(dataset.eval_dataset, dict):
                        for k, v in dataset.eval_dataset.items():
                            log_dataset(f"Eval {k}", v)
                    else:
                        log_dataset("Eval", dataset.eval_dataset)
                    print("-------------------\n")
                for dataset in datasets:
                    log_samples("Train", dataset.train_dataset)
                    log_samples("Test", dataset.test_dataset)
                    if isinstance(dataset.eval_dataset, dict):
                        for k, v in dataset.eval_dataset.items():
                            log_samples(f"Eval {k}", v)
                    else:
                        log_samples("Eval", dataset.eval_dataset)
                    print("-------------------\n")

                print("Hybrid dataset:")
                log_dataset("Train", self.train_dataset)
                log_dataset("Test", self.test_dataset)
                if isinstance(dataset.eval_dataset, dict):
                    for k, v in dataset.eval_dataset.items():
                        log_dataset(f"Eval {k}", v)
                else:
                    log_dataset("Eval", dataset.eval_dataset)
                log_samples("Train", self.train_dataset)
                log_samples("Test", self.test_dataset)
                if isinstance(dataset.eval_dataset, dict):
                    for k, v in dataset.eval_dataset.items():
                        log_samples(f"Eval {k}", v)
                else:
                    log_samples("Eval", dataset.eval_dataset)

    def concat_datasets(self, split, to_concatenate):
        try:
            dataset = concatenate_datasets(to_concatenate, split=split)
        except Exception as e:
            print(f"\nError concatenating datasets for split {split}: {e}")
            for i in range(len(to_concatenate)):
                for audio_key in [
                    "audio",
                    "audio_output",
                    "audio_input",
                    "audio_condition",
                ]:
                    if audio_key in to_concatenate[i].column_names:
                        print(
                            f"Trying to align Audio columns for split {split}"
                        )
                        to_concatenate[i] = to_concatenate[i].cast_column(
                            audio_key, Audio(decode=False)
                        )
                if "video" in to_concatenate[i].column_names:
                    to_concatenate[i] = to_concatenate[i].cast_column(
                        "video",
                        Video(decode=False),
                        num_proc=self.data_args.num_proc_for_preprocessing,
                    )
            dataset = concatenate_datasets(to_concatenate, split=split)
            return dataset
        return dataset


if __name__ == "__main__":
    config_path = (
        f"{os.getenv('SPEECHLMM_ROOT')}/speechlmm/dataset/example.yml"
    )
    data_args = DataArguments(
        sampling_rate=16000,
        data_config_path=config_path,
        is_multimodal=True,
        dataloader_debug=True,
        organize_eval_dataset_per_task=True,
        filter_broken_samples=False,
        rebuild_dataset_cache=False,
    )
    datasets = DatasetsWrapper(data_args)
    train_dataset = datasets.train_dataset
