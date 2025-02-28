import pickle
from pathlib import Path
from typing import Optional

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import CustomAudioDataset
from speechlmm.dataset.custom_dataset.preparers import (
    SpeechToSpeechTranslationPreparer,
)


class CvssDataset(CustomAudioDataset):
    name = "CVSS"
    codename = "cvss"

    sequence_keys = ["audio_input", "audio_output"]
    duration_keys = ["duration_input", "duration_output"]
    text_keys = ["text_input", "text_output"]
    token_rate_validation_pairs = [
        ("duration_input", "text_input", 700),
        ("duration_output", "text_output", 700),
    ]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.preparers = {"S2ST": SpeechToSpeechTranslationPreparer()}
        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )

    # def _load_dataset(
    #     self,
    #     source_language: str,
    #     target_language: str,
    #     partition_name: str,
    #     partition_spec: dict,
    # ):
    #     dataset = super()._load_dataset(
    #         source_language=source_language,
    #         target_language=target_language,
    #         partition_name=partition_name,
    #         partition_spec=partition_spec,
    #     )

    #     def decode_audio(example):
    #         example["audio_input"] = pickle.loads(example["audio_input"])
    #         example["audio_output"] = pickle.loads(example["audio_output"])
    #         return example

    #     dataset = dataset.map(
    #         decode_audio,
    #         num_proc=self.num_proc_for_preprocessing,
    #         desc="decoding audios",
    #     )
    #     return dataset

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        # path_suffix = Path(
        #     f"CVSS_aligned_{source_language}_{target_language}",
        #     f"{partition_name}.parquet",
        # )
        path_suffix = Path(
            f"{source_language}-{target_language}",
            f"{partition_name}.parquet",
        )
        dataset_path = Path(dataset_dir, path_suffix)
        return str(dataset_path)
