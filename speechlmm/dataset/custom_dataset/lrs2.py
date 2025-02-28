import logging
from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomVideoDataset
from speechlmm.dataset.custom_dataset.preparers import (
    VisualSpeechRecognitionPreparer,
)

logger = logging.getLogger(__name__)


class LRS2VideoOnlyDataset(CustomVideoDataset):
    name = "LRS2VideoOnly"
    codename = "lrs2_video_only"

    duration_keys = ["duration"]
    text_keys = ["transcription"]
    token_rate_validation_triplets = [("duration", "transcription", 700)]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.preparers = {"VSR": VisualSpeechRecognitionPreparer()}
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
        path_suffix = Path(f"{partition_name}.parquet")
        dataset_path = Path(dataset_dir, path_suffix)
        return str(dataset_path)
