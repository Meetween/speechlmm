from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import NUM_WORDS_FILTER, CustomDataset
from speechlmm.dataset.custom_dataset.preparers import TextInstructPreparer


class EverythingInstructMultilingualDataset(CustomDataset):
    name = "EverythingInstructMultilingualDataset"
    codename = "everything_instruct_multilingual"

    optional_filters = [NUM_WORDS_FILTER]
    text_keys = ["instruction", "input", "output"]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.preparers = {"TextInstruct": TextInstructPreparer()}
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
        filename = f"{partition_name}.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)
