from pathlib import Path
from typing import Optional

from datasets import concatenate_datasets

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomAudioDataset
from speechlmm.dataset.custom_dataset.preparers import (
    MachineTranslationPreparer,
    SpeechRecognitionPreparer,
    SpeechTranslationPreparer,
)


class EuroparlDataset(CustomAudioDataset):
    name = "Europarl"
    codename = "europarl"

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
        self.supported_language_directions = {
            "de-en",
            "de-es",
            "de-fr",
            "de-it",
            "en-de",
            "en-es",
            "en-fr",
            "en-it",
            "es-de",
            "es-en",
            "es-fr",
            "es-it",
            "fr-de",
            "fr-en",
            "fr-es",
            "fr-it",
            "it-de",
            "it-en",
            "it-es",
            "it-fr",
        }
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(),
            "ST": SpeechTranslationPreparer(),
            "MT": MachineTranslationPreparer(
                remap_keys={
                    "transcription": "text_input",
                    "translation": "text_output",
                }
            ),
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
        language_direction = f"{source_language}_{target_language}"

        filename = f"{language_direction}/{partition_name}.parquet"

        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)
