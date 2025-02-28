import logging
import random
from collections import defaultdict

import numpy as np
from datasets import Audio, DownloadMode, load_dataset
from torch.utils.data import Dataset

from speechlmm.dataset.config import CustomDatasetConfig


class Refactoring(Dataset):
    def __init__(
        self,
        config: CustomDatasetConfig,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        partitions = config.partitions
        # check if the task is supported
        # FIXME specify the supported tasks for the dataset
        assert config.task in ["ASR"], NotImplementedError(
            "Only ASR ans ST task is supported for CoVoST dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.validation_dataset = (
            None,
            None,
            None,
        )

        preprocess_fn = getattr(self, "cast_to_audio_dataset")

        # FIXME customize this at your best convenience to adapt to your dataset partitions
        # for language in tqdm(config.languages, desc="Loading dataset"):
        #     print(f"Loading {language} dataset")
        for split, info in partitions.items():
            print(f"Loading {split} dataset")
            dataset = load_dataset(
                "parquet",
                data_files={
                    # FIXME customize this to adapt to your dataset path
                    f"{split.replace('-', '_')}": f"{config.datapath}/{split}/data.parquet",
                },
                split=f"{split.replace('-', '_')}[{info['amount']}]",
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if rebuild_cache
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )
            dataset = dataset.map(
                lambda example: preprocess_fn(example),
                batched=False,
            )

            casted = dataset.cast_column("audio", Audio())
            casted.to_parquet(
                f"{config.datapath}_refactoring/{split}/data.parquet",
                batch_size=512,
            )

    def __len__(self):
        return len(self.train_dataset)

    def cast_to_audio_dataset(self, example):
        example["audio"] = {
            "path": example["path"],
            "sampling_rate": example["sr"],
            "array": np.array(example["audio"]),
        }
        del example["path"]
        del example["sr"]
        return example
