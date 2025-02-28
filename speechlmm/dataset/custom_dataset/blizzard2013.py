import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import DownloadMode, load_dataset
from joblib import Parallel, delayed

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import (
    CaptionMixin,
    ConditioningMixin,
    CustomAudioDataset,
)
from speechlmm.dataset.custom_dataset.preparers import (
    SpeechRecognitionPreparer,
    TextToSpeechPreparer,
)

logger = logging.getLogger(__name__)


class BlizzardChallenge2013Dataset(CustomAudioDataset):
    name = "Blizzard Challenge 2013"
    codename = "blizzard2013"

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        remap_keys = {"text": "transcription"}
        remap_keys_tts = {"audio": "audio_output", **remap_keys}
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(remap_keys=remap_keys),
            "TTS": TextToSpeechPreparer(
                remap_keys=remap_keys_tts, dataset=self
            ),
        }

        if config.languages != ["en"]:
            logger.warning(
                f"Only English is supported for Blizzard Challenge 2013. Other languages will be ignored."
            )

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
        dataset_path = Path(dataset_dir, f"BC2013_v0.{partition_name}.parquet")
        return str(dataset_path)
