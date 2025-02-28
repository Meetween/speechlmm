import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from datasets import DownloadMode, load_dataset

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import (
    ConditioningMixin,
    CustomAudioDataset,
)
from speechlmm.dataset.custom_dataset.preparers import (
    TextToSpeechCaptionOnlyPreparer,
    TextToSpeechPreparer,
)

logger = logging.getLogger(__name__)


class DapsFloresDataset(CustomAudioDataset, ConditioningMixin):
    name = "DAPSFlores"
    codename = "dapsflores"

    audio_keys = ["audio"]
    duration_keys = []
    token_rate_validation_triplets = []

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        remap_keys = {}
        remap_keys_tts = {"audio": "audio_output", **remap_keys}
        caption_only = config.additional_args.get("caption_only", False)

        self.preparers = {
            "TTS": (
                TextToSpeechPreparer(remap_keys=remap_keys_tts, dataset=self)
                if not caption_only
                else TextToSpeechCaptionOnlyPreparer(
                    remap_keys=remap_keys_tts, dataset=self
                )
            ),
        }

        if config.languages != ["en"]:
            logger.warning(
                f"Only English is supported for DAPS. Other languages will be ignored."
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
        filename = "dapsflores_240_en_clean.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)

    def get_conditioning_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        return {"audio": example["audio_output"]}
