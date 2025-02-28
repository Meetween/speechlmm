from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomDataset
from speechlmm.dataset.custom_dataset.preparers import MachineTranslationPreparer


class NllbParacrawlDataset(CustomDataset):
    name = "NllbParacrawl"
    codename = "nllbparacrawl"

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.supported_language_directions = {
            "de-en",
            "de-es",
            "de-fr",
            "de-it",
            "en-es",
            "en-fr",
            "en-it",
            "es-fr",
            "es-it",
            "fr-it",
        }
        self.preparers = {
            "MT": MachineTranslationPreparer(
                remap_keys={
                    "source": "text_input",
                    "target": "text_output",
                }
            )
        }
        if config.languages[0] == "all":
            config.languages = [
                direction for direction in self.supported_language_directions
            ]
        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        language_direction = f"{source_language}-{target_language}"
        if language_direction in self.supported_language_directions:
            filename = f"{language_direction}/{partition_name}.parquet"
        else:
            reverse_language_direction = f"{target_language}-{source_language}"
            filename = f"{reverse_language_direction}/{partition_name}.parquet"

        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)

    def _load_dataset(
        self,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ):
        language_direction = f"{source_language}-{target_language}"
        if language_direction in self.supported_language_directions:
            return super()._load_dataset(
                source_language=source_language,
                target_language=target_language,
                partition_name=partition_name,
                partition_spec=partition_spec,
            )

        reverse_language_direction = f"{target_language}-{source_language}"
        if (
            reverse_language_direction
            not in self.supported_language_directions
        ):
            raise ValueError(
                f"Language pair <{source_language}, {target_language}> (in "
                f"either direction) is not supported in {self.name}."
            )

        dataset = super()._load_dataset(
            source_language=target_language,  # swapped
            target_language=source_language,  # swapped
            partition_name=partition_name,
            partition_spec=partition_spec,
        )

        # swap source and target language columns
        # fmt: off
        dataset = (
            dataset
            .rename_column("source", "source_")
            .rename_column("target", "source")
            .rename_column("source_", "target")
        )
        # fmt: on
        return dataset
