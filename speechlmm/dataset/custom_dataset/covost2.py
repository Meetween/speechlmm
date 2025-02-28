from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomAudioDataset
from speechlmm.dataset.custom_dataset.preparers import (
    SpeechRecognitionPreparer,
    SpeechTranslationPreparer,
)


class Covost2Dataset(CustomAudioDataset):
    name = "CoVoST 2"
    codename = "covost2"

    sequence_keys = ["audio"]
    duration_keys = ["duration"]
    text_keys = ["transcription", "translation"]
    token_rate_validation_triplets = [
        ("duration", "transcription", 700),
        ("duration", "translation", 700),
    ]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(),
            "ST": SpeechTranslationPreparer(),
        }
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
        filename = f"covost2.{source_language}_{target_language}.{partition_name}.parquet"
        dataset_path = Path(dataset_dir, filename)
        if not dataset_path.exists():
            subdir = "en_xx" if source_language == "en" else "xx_en"
            dataset_path = Path(dataset_dir, subdir, filename)

        return str(dataset_path)
