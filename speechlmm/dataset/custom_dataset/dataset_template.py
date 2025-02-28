import random
from collections import defaultdict

from datasets import DownloadMode, concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from speechlmm.dataset.config import CustomDatasetConfig


class MyAwesomeCustomModel(Dataset):
    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        DEFAULT_AUDIO_TOKEN: str = "<audio>",
        rebuild_cache: bool = False,
    ):
        super().__init__()
        self.config = config
        partitions = config.partitions
        # check if the task is supported
        # FIXME specify the supported tasks for the dataset
        assert config.task in ["TASK1", "TASK2"], NotImplementedError(
            "Only ASR ans ST task is supported for CoVoST dataset"
        )

        # get train, test and validation datasets
        datasets = defaultdict(list)
        self.train_dataset, self.test_dataset, self.validation_dataset = (
            None,
            None,
            None,
        )

        self.DEFAULT_AUDIO_TOKEN = DEFAULT_AUDIO_TOKEN

        preprocess_fn = getattr(self, f"preprocess_{config.task}")

        # FIXME customize this at your best convenience to adapt to your dataset partitions
        for language in tqdm(config.languages, desc="Loading dataset"):
            print(f"Loading {language} dataset")
            for split, info in partitions.items():
                print(f"Loading {split} dataset")
                dataset = load_dataset(
                    "parquet",
                    data_files={
                        # FIXME customize this to adapt to your dataset path
                        f"{split}": f"{config.datapath}/path/to_{language}.{split}.parquet",
                    },
                    split=f"{split}[{info['amount']}]",
                    download_mode=(
                        DownloadMode.FORCE_REDOWNLOAD
                        if rebuild_cache
                        else DownloadMode.REUSE_DATASET_IF_EXISTS
                    ),
                )
                dataset = dataset.map(
                    lambda example: preprocess_fn(example, language),
                    batched=False,
                )
                datasets[info["destination"]].append(dataset)

        for destination, dataset in datasets.items():
            print(f"Concatenating {destination} dataset")
            if len(dataset) == 1:
                setattr(self, f"{destination}_dataset", dataset[0])
            elif len(dataset):
                setattr(
                    self,
                    f"{destination}_dataset",
                    concatenate_datasets(dataset),
                )

    def __len__(self):
        return len(self.train_dataset)

    def preprocess_ASR(self, example, language: str):
        example["language"] = language
        question, answer = "", ""
        target = language
        # NOTE harcoded cause not implemented yet
        target = "en"
        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][target]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][target]
            )
        example["source"] = [
            {
                "from": "human",
                "value": question + f"{self.DEFAULT_AUDIO_TOKEN}\n",
            },
            # FIXME customize this to adapt to your dataset
            {"from": "gpt", "value": answer + example["value"]},
        ]

        return example

    def preprocess_ST(self, example, language: str):
        example["language"] = language
        question, answer = "", ""
        target = language
        # NOTE harcoded cause not implemented yet
        target = "en"
        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][target]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][target]
            )
        example["source"] = [
            {
                "from": "human",
                "value": question + f"{self.DEFAULT_AUDIO_TOKEN}\n",
            },
            # FIXME customize this to adapt to your dataset
            {"from": "gpt", "value": answer + example["value"]},
        ]

        return example

    def preprocess_SLU(self, example, language: str):
        example["language"] = language
        question, answer = "", ""
        target = language
        # NOTE harcoded cause not implemented yet
        target = "en"
        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][target]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][target]
            )
        example["source"] = [
            {
                "from": "human",
                "value": question + f"{self.DEFAULT_AUDIO_TOKEN}\n",
            },
            # FIXME customize this to adapt to your dataset
            {"from": "gpt", "value": answer + example["value"]},
        ]

        return example

    def preprocess_NEGATIVE(self, example, language: str):
        example["language"] = language
        question, answer = "", ""
        target = language
        # NOTE harcoded cause not implemented yet
        target = "en"
        if self.config.INPUTS_TEXT_LIST is not None:
            question = random.choice(
                self.config.INPUTS_TEXT_LIST[self.config.task][target]
            )
        if self.config.OUTPUTS_TEXT_LIST is not None:
            answer = random.choice(
                self.config.OUTPUTS_TEXT_LIST[self.config.task][target]
            )
        example["source"] = [
            {
                "from": "human",
                "value": question + f"{self.DEFAULT_AUDIO_TOKEN}\n",
            },
            # FIXME customize this to adapt to your dataset
            {"from": "gpt", "value": answer + example["value"]},
        ]

        return example
