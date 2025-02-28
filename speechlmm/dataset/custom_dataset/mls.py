from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomAudioDataset
from speechlmm.dataset.custom_dataset.preparers import SpeechRecognitionPreparer


class MlsDataset(CustomAudioDataset):
    name = "Multilingual LibriSpeech"
    codename = "mls"

    sequence_keys = ["audio"]
    duration_keys = ["duration"]
    text_keys = ["transcript"]
    token_rate_validation_triplets = [("duration", "transcript", 700)]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        remap_keys = {"transcript": "transcription"}
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(remap_keys=remap_keys),
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
        filename = f"{source_language}/{partition_name}.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)
