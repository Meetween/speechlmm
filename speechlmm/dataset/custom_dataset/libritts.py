import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import DownloadMode, Value, load_dataset
from joblib import Parallel, delayed

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import (
    NUM_WORDS_FILTER,
    CaptionMixin,
    ConditioningMixin,
    CustomAudioDataset,
    get_audio_duration,
)
from speechlmm.dataset.custom_dataset.preparers import (
    InterleavedTextAudioNTPPreparer,
    SpeechRecognitionPreparer,
    SpeechToSpeechTranslationPreparer,
    TextToSpeechBasePreparer,
    TextToSpeechCaptionOnlyPreparer,
    TextToSpeechPreparer,
)

logger = logging.getLogger(__name__)


class LibriTtsDataset(CustomAudioDataset, ConditioningMixin, CaptionMixin):
    name = "LibriTTS"
    codename = "libritts"

    sequence_keys = ["audio"]
    duration_keys = ["duration"]
    text_keys = ["text_normalized"]
    token_rate_validation_triplets = [("duration", "text_normalized", 700)]
    speaker_id_key = "speaker_id"

    columns_to_cast = {"speaker_id": Value(dtype="string", id=None)}

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.config = config
        remap_keys = {"text_normalized": "transcription"}
        if (
            config.task == "TTS"
            or config.task == "TTSBase"
            or config.task == "InterleavedTextAudioNTP"
        ):
            remap_keys = {"audio": "audio_output", **remap_keys}
        elif config.task == "S2ST":
            self.sequence_keys = ["audio_input", "audio_output"]
            self.duration_keys = ["duration_input", "duration_output"]
            self.token_rate_validation_triplets = [
                ("duration_input", "text_input", 700),
                ("duration_output", "text_output", 700),
            ]
            self.text_keys = ["text_input", "text_output"]
            self.speaker_id_key = "speaker_id_output"
            remap_keys = {"speaker_id_output": "speaker_id", **remap_keys}
            self.duration_getters.update(
                {
                    "audio_input": get_audio_duration,
                    "audio_output": get_audio_duration,
                }
            )

        self.remap_keys = remap_keys

        caption_only = config.additional_args.get("caption_only", False)

        if config.task == "InterleavedTextAudioNTP":
            self.optional_filters = [NUM_WORDS_FILTER]

        self.preparers = {
            "ASR": SpeechRecognitionPreparer(remap_keys=remap_keys),
            "TTS": (
                TextToSpeechPreparer(remap_keys=remap_keys, dataset=self)
                if not caption_only
                else TextToSpeechCaptionOnlyPreparer(
                    remap_keys=remap_keys, dataset=self
                )
            ),
            "TTSBase": TextToSpeechBasePreparer(
                remap_keys=remap_keys, dataset=self
            ),
            "InterleavedTextAudioNTP": InterleavedTextAudioNTPPreparer(
                remap_keys=remap_keys, dataset=self
            ),
            "S2ST": SpeechToSpeechTranslationPreparer(
                remap_keys=remap_keys, dataset=self
            ),
        }

        self.is_libritts_r_filtered = "LibriTTS-R-Filtered" in config.datapath
        self.add_speaker_descriptions = config.additional_args.get(
            "add_speaker_descriptions", False
        )
        if config.languages != ["en"]:
            logger.warning(
                f"Only English is supported for LibriTTS. Other languages will be ignored."
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

        if self.is_libritts_r_filtered and self.add_speaker_descriptions:
            self._load_utterance_descriptions_for_partition(partition_name)

        if (
            self.config.task == "TTS" or self.config.task == "S2ST"
        ) and not self.config.additional_args.get("caption_only", False):
            self._build_speakers_map()

    def _load_utterance_descriptions_for_partition(self, partition_name):
        if "LibriTTS-R" in self.config.datapath:
            actual_partition_name = partition_name.replace("_", ".")
        else:
            actual_partition_name = partition_name.replace("_", "-")
        speaker_descriptions_path = f"{self.config.datapath}-Speaker-Descriptions/{actual_partition_name}.parquet"
        speaker_descriptions_dataset = load_dataset(
            "parquet",
            data_files={
                f"{partition_name}_speaker_description": speaker_descriptions_path,
            },
            split=f"{partition_name}_speaker_description",
            download_mode=(
                DownloadMode.FORCE_REDOWNLOAD
                if self.rebuild_cache
                else DownloadMode.REUSE_DATASET_IF_EXISTS
            ),
        )
        # Convert the dataset to a dictionary for faster lookups
        self.cur_utterance_descriptions = dict(
            zip(
                speaker_descriptions_dataset["id"],
                speaker_descriptions_dataset["text_description"],
            )
        )

    def _build_speakers_map(self):
        if self.config.additional_args.get("condition_on_input", False):
            self.cur_speakers_map = None
            return

        logger.info("Building speakers map")
        speakers_map = defaultdict(list)

        def add_to_speakers_map(idx, example):
            speakers_map[example[self.speaker_id_key]].append(idx)

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
        if "LibriTTS-R" in dataset_dir:
            stem = partition_name.replace("_", ".")
        elif "translated" in dataset_dir:
            stem = partition_name
        else:
            actual_partition_name = partition_name.replace("_", "-")
            stem = f"{actual_partition_name}/data"
        filename = f"{stem}.parquet"
        dataset_path = Path(dataset_dir, filename)
        return str(dataset_path)

    def get_conditioning_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self.config.additional_args.get("condition_on_input", False):
            example["audio"] = example["audio_input"]
            return example

        candidate_examples_ids = self.cur_speakers_map[example["speaker_id"]]
        assert len(candidate_examples_ids) > 1, "Speaker has < 2 samples"

        found_valid_conditioning = False
        while not found_valid_conditioning:
            candidate_example_id = random.choice(candidate_examples_ids)
            candidate_example = self.cur_dataset[candidate_example_id]
            if candidate_example != example:
                found_valid_conditioning = True

        if not "audio" in candidate_example:
            candidate_example["audio"] = candidate_example["audio_output"]

        return candidate_example

    def get_caption_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        if not self.add_speaker_descriptions:
            return None

        return self.cur_utterance_descriptions[example["id"]]
