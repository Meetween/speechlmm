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
    TextToSpeechCaptionOnlyPreparer,
    TextToSpeechPreparer,
)

logger = logging.getLogger(__name__)


class CommonVoiceDataset(CustomAudioDataset, ConditioningMixin, CaptionMixin):
    name = "CommonVoice"
    codename = "common_voice"

    sequence_keys = ["audio"]
    duration_keys = ["duration"]
    text_keys = ["sentence"]
    token_rate_validation_triplets = [("duration", "sentence", 700)]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        remap_keys = {"sentence": "transcription", "client_id": "speaker_id"}
        remap_keys_tts = {"audio": "audio_output", **remap_keys}
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(remap_keys=remap_keys),
            "TTS": TextToSpeechPreparer(
                remap_keys=remap_keys_tts, dataset=self
            ),
        }

        # self.is_libritts_r_filtered = "LibriTTS-R-Filtered" in config.datapath
        self.add_speaker_descriptions = config.additional_args.get(
            "add_speaker_descriptions", False
        )
        #
        supported_languages = ["en", "de", "fr", "it", "es"]
        for language in config.languages:
            if language not in supported_languages:
                logger.warning(
                    f"Language {language} is not supported for {self.name}. "
                    f"Supported languages are {supported_languages}."
                )

        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )

    def _post_filter_hook(
        self,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ):
        super()._post_filter_hook(
            source_language=source_language,
            target_language=target_language,
            partition_name=partition_name,
            partition_spec=partition_spec,
        )

        if self.config.task == "TTS":
            self._build_speakers_map()

    def _build_speakers_map(self):
        speakers_map = defaultdict(list)

        def add_to_speakers_map(idx, example):
            speakers_map[example["client_id"]].append(idx)

        speakers_to_pop = []
        valid_sample_indices = []

        def update_speakers_map_and_valid_indices(speaker, min_samples=2):
            sample_indices = speakers_map[speaker]
            if len(sample_indices) < min_samples:
                speakers_to_pop.append(speaker)
            else:
                valid_sample_indices.extend(sample_indices)

        with Parallel(
            n_jobs=self.num_proc_for_preprocessing,
            require="sharedmem",
            verbose=10,
        ) as parallel:
            parallel(
                delayed(add_to_speakers_map)(idx, example)
                for idx, example in enumerate(self.cur_dataset)
            )

            # Filter out speakers with < 2 samples
            parallel(
                delayed(update_speakers_map_and_valid_indices)(
                    speaker, min_samples=2
                )
                for speaker in speakers_map
            )

        for speaker in speakers_to_pop:
            speakers_map.pop(speaker)

        # Update dataset to remove entries of speakers with < 2 samples
        initial_length = len(self.cur_dataset)
        self.cur_dataset = self.cur_dataset.select(
            indices=valid_sample_indices,
        )
        num_removed_samples = initial_length - len(self.cur_dataset)
        if num_removed_samples > 0:
            # We must rebuild the speakers map because the dataset has been
            # filtered, which means the sample indices have changed
            speakers_map = defaultdict(list)
            with Parallel(
                n_jobs=self.num_proc_for_preprocessing,
                require="sharedmem",
                verbose=10,
            ) as parallel:
                parallel(
                    delayed(add_to_speakers_map)(idx, example)
                    for idx, example in enumerate(self.cur_dataset)
                )

        self.cur_speakers_map = speakers_map

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        stem = source_language
        filename = f"{stem}/{partition_name}.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)

    def get_conditioning_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        candidate_examples_ids = self.cur_speakers_map[example["speaker_id"]]
        assert len(candidate_examples_ids) > 1, "Speaker has < 2 samples"

        found_valid_conditioning = False
        while not found_valid_conditioning:
            candidate_example_id = random.choice(candidate_examples_ids)
            candidate_example = self.cur_dataset[candidate_example_id]
            if candidate_example["sentence"] != example["transcription"]:
                found_valid_conditioning = True

        return candidate_example

    def get_caption_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        if not self.add_speaker_descriptions:
            return None

        return self.cur_utterance_descriptions[example["id"]]


if __name__ == "__main__":
    import os

    from speechlmm.arguments import DataArguments
    from speechlmm.dataset.datasets_wrapper import DatasetsWrapper

    TEST_ROOT_PATH = f"{os.getenv('DATA_HOME')}/common_voice_17_0"
    TEST_CONFIG_PATH = (
        f"{os.getenv('SPEECHLMM_ROOT')}/conf/datasets/commonvoice.yml"
    )

    data_args = DataArguments(
        sampling_rate=16000,
        data_config_path=TEST_CONFIG_PATH,
        is_multimodal=True,
        dataloader_debug=False,
        organize_eval_dataset_per_task=True,
        rebuild_dataset_cache=False,
    )

    all_datasets = DatasetsWrapper(data_args)
