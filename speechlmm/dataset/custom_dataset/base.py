import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from datasets import (
    Audio,
    DownloadMode,
    Value,
    Video,
    concatenate_datasets,
    load_dataset,
)

from decord import VideoReader
from tqdm import tqdm

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.utils import (
    get_dict_fingerprint,
    load_parquet_shards,
    save_parquet_shards,
)

logger = logging.getLogger(__name__)


CTC_ALIGNER_FILTER = "ctc_aligner"
NUM_WORDS_FILTER = "num_words"
QUALITY_FILTER = "quality"
SPAM_FILTER = "spam"


class CustomDataset(metaclass=ABCMeta):
    name = ""
    codename = ""
    splits = ["train", "eval", "test"]

    text_keys = ["text"]

    optional_filters = []

    columns_to_cast = dict()
    preparers = dict()

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.cache_final_datasets = cache_final_datasets
        self.rebuild_cache = rebuild_cache
        self.num_proc_for_preprocessing = num_proc_for_preprocessing

        if len(self.name) == 0:
            raise ValueError(
                f"Must define a name for {self.__class__.__name__}"
            )
        if len(self.codename) == 0:
            raise ValueError(
                f"Must define a codename for {self.__class__.__name__}"
            )

        if config.task not in self.preparers:
            raise ValueError(
                f"You must define a prepare function for task '{config.task}'."
            )

        self.fingerprint = get_dict_fingerprint(asdict(config))

        for split in self.splits:
            setattr(self, f"{split}_dataset", None)

        all_destinations = set(
            partition_spec["destination"]
            for partition_spec in config.partitions.values()
        )
        assert all(
            d in self.splits for d in all_destinations
        ), "Invalid destination in partitions."

        for destination in all_destinations:
            self._maybe_load_cached_dataset(destination)

        partitions_per_destination = defaultdict(list)
        for partition_name, partition_spec in config.partitions.items():
            destination_split = partition_spec["destination"]
            if (
                getattr(self, f"{destination_split}_dataset") is not None
                and not self.rebuild_cache
            ):
                # dataset already loaded from cache
                continue

            partition = self._load_and_preprocess_partition(
                partition_name, partition_spec
            )
            partitions_per_destination[destination_split].append(partition)

        # NOTE: if the datasets were loaded from cache,
        # `partitions_per_destination` will be empty and the following
        # loop will be skipped altogether
        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )
            if self.cache_final_datasets:
                self._cache_dataset(destination)

    def _filter_example(
        self, example, language, partition_name, partition_spec
    ) -> bool:
        if NUM_WORDS_FILTER in self.optional_filters:
            return self._filter_by_num_words(
                example,
                min_num_words=partition_spec.get("min_num_words", None),
                max_num_words=partition_spec.get("max_num_words", None),
            )

        return True

    def _filter_by_num_words(
        self,
        example: Dict[str, Any],
        min_num_words: int,
        max_num_words: int,
    ) -> bool:
        """
        Filter examples based on number of words in the text.

        Args:
            example: The example to check
            min_num_words: Minimum number of words required (-1 for no minimum)
            max_num_words: Maximum number of words allowed (-1 for no maximum)

        Returns:
            bool: True if example passes the filter, False otherwise
        """
        # Check if text_keys exists and has valid content
        if not hasattr(self, "text_keys") or not self.text_keys:
            logging.warning("text_keys not defined or empty in dataset class")
            return True

        # Check all text fields meet the criteria
        for text_key in self.text_keys:
            if text_key not in example:
                logging.warning(f"{text_key} not found in example")
                continue

            text = example[text_key]
            if text is None or not isinstance(text, str):
                logging.warning(f"Invalid text value for {text_key}: {text}")
                return False

            # Count words (splitting on whitespace)
            num_words = len(text.split())

            # Check minimum words if specified
            if min_num_words is not None and num_words < min_num_words:
                return False
            # Check maximum words if specified
            if max_num_words is not None and num_words > max_num_words:
                return False

        return True

    def _maybe_load_cached_dataset(self, destination: str):
        cached_dataset_dir = self._get_cached_dataset_dir(destination)
        if cached_dataset_dir.exists() and not self.rebuild_cache:
            setattr(
                self,
                f"{destination}_dataset",
                load_parquet_shards(
                    cached_dataset_dir,
                    num_proc=self.num_proc_for_preprocessing,
                ),
            )

    def _cache_dataset(self, destination: str):
        cached_dataset_dir = self._get_cached_dataset_dir(destination)
        if cached_dataset_dir.exists() and not self.rebuild_cache:
            raise ValueError(
                f"Dataset is already cached at {cached_dataset_dir}"
            )

        dataset_to_save = getattr(self, f"{destination}_dataset")
        if dataset_to_save is None:
            raise ValueError(
                f"`self.{destination}_dataset` is None, can't cache it."
            )

        cached_dataset_dir.mkdir(parents=True, exist_ok=True)
        save_parquet_shards(
            dataset_to_save,
            cached_dataset_dir,
            max_shard_size="1GB",
            num_proc=self.num_proc_for_preprocessing,
        )

    def _get_cached_dataset_dir(self, destination: str):
        return Path(
            datasets.config.HF_DATASETS_CACHE,
            f"{self.codename}_{self.config.task}_{self.fingerprint}",
            f"{destination}",
        )

    def _load_and_preprocess_partition(
        self, partition_name: str, partition_spec: dict
    ):
        datasets_to_concatenate = []
        for language in tqdm(
            self.config.languages, desc=f"loading dataset '{self.name}'"
        ):
            source_language, *rest = language.split("-")
            target_language = rest[0] if len(rest) > 0 else None

            self.cur_dataset = self._load_dataset(
                source_language,
                target_language,
                partition_name,
                partition_spec,
            )

            filter_fn = lambda example: self._filter_example(
                example,
                language=language,
                partition_name=partition_name,
                partition_spec=partition_spec,
            )
            logging.info(
                f"Dataset size before filtering: {len(self.cur_dataset)}"
            )
            self.cur_dataset = self.cur_dataset.filter(
                filter_fn,
                num_proc=self.num_proc_for_preprocessing,
                desc="filtering bad examples",
            )
            logging.info(
                f"Dataset size after filtering: {len(self.cur_dataset)}"
            )

            self._post_filter_hook(
                source_language,
                target_language,
                partition_name,
                partition_spec,
            )

            task_preparer = self.preparers[self.config.task]
            self.cur_dataset = self.cur_dataset.map(
                lambda example: task_preparer.prepare_example(
                    example,
                    source_language=source_language,
                    target_language=target_language,
                    config=self.config,
                ),
                num_proc=self.num_proc_for_preprocessing,
                desc=f"preprocessing ({self.config.task})",
            )

            self.cur_dataset = self._cast_columns(self.cur_dataset)

            datasets_to_concatenate.append(self.cur_dataset)

        if len(datasets_to_concatenate) == 0:
            raise ValueError(
                f"No data to gather from partition '{partition_name}'."
            )

        logger.info(
            f"Gathering data across languages {self.config.languages} "
            f"from partition '{partition_name}'"
        )
        return concatenate_datasets(datasets_to_concatenate)

    def _load_dataset(
        self,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ):
        dataset_path = self._get_dataset_path(
            dataset_dir=self.config.datapath,
            source_language=source_language,
            target_language=target_language,
            partition_name=partition_name,
            partition_spec=partition_spec,
        )

        logger.info(
            f"loading dataset: {source_language} -> {target_language} ({partition_name})"
        )
        dataset = load_dataset(
            "parquet",
            data_files={partition_name: dataset_path},
            split=f"{partition_name}[{partition_spec['amount']}]",
            download_mode=(
                DownloadMode.FORCE_REDOWNLOAD
                if self.rebuild_cache
                else DownloadMode.REUSE_DATASET_IF_EXISTS
            ),
            num_proc=self.num_proc_for_preprocessing,
        )

        return dataset

    @abstractmethod
    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        raise NotImplementedError

    def _cast_columns(self, dataset: datasets.Dataset):
        if len(self.columns_to_cast) == 0:
            return dataset

        if all(
            column_name not in dataset.features
            for column_name in self.columns_to_cast
        ):
            return dataset

        new_features = dataset.features.copy()
        for column_name, column_type in self.columns_to_cast.items():
            if column_name in dataset.features:
                new_features[column_name] = column_type

        return dataset.cast(
            new_features, num_proc=self.num_proc_for_preprocessing
        )

    def _post_filter_hook(
        self,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ):
        pass  # no-op by default

    def __len__(self):
        return len(self.train_dataset)


class CustomTemporalDataset(CustomDataset):
    sequence_keys = ["sequence"]
    duration_keys = ["duration"]
    text_keys = ["text"]

    # (duration_key, text_keys, max_tokens_per_minute)
    token_rate_validation_triplets = [("duration", "text", 700)]

    optional_filters = [CTC_ALIGNER_FILTER, NUM_WORDS_FILTER]

    duration_getters = dict()

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        num_sequence_keys = len(self.sequence_keys)
        if num_sequence_keys <= 0:
            raise ValueError("`sequence_keys` must be non-empty")

        num_text_keys = len(self.text_keys)
        if num_text_keys <= 0:
            raise ValueError("`text_keys` must be non-empty")

        if num_text_keys != num_sequence_keys:
            if num_text_keys == 1:
                self.text_keys = [self.text_keys[0]] * num_sequence_keys
            elif num_sequence_keys == 1:
                self.sequence_keys = [self.sequence_keys[0]] * num_text_keys
            else:
                raise ValueError(
                    "`sequence_keys` and `text_keys` must have the same "
                    "length. Alternatively, one of them can have length 1, "
                    "in which case a sort of broadcasting will be applied."
                )

        num_duration_keys = len(self.duration_keys)
        if num_duration_keys != num_sequence_keys:
            if num_duration_keys == 1:
                self.duration_keys = [
                    self.duration_keys[0]
                ] * num_sequence_keys
            else:
                raise ValueError(
                    "`sequence_keys` and `duration_keys` must have the same "
                    "length. Alternatively, `duration_keys` can be a single "
                    "key, in which case it will be applied to all sequence "
                    "keys."
                )

        for sequence_key in self.sequence_keys:
            if sequence_key not in self.duration_getters:
                raise ValueError(
                    f"Duration getter for sequence key '{sequence_key}' "
                    f"was not defined."
                )

        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )

    def _filter_example(
        self, example, language, partition_name, partition_spec
    ) -> bool:
        # First call parent class filter (which handles num_words check)
        if not super()._filter_example(
            example, language, partition_name, partition_spec
        ):
            return False

        is_duration_within_limits = self._filter_by_duration(
            example,
            min_duration=partition_spec.get("min_duration", None),
            max_duration=partition_spec.get("max_duration", None),
        )
        is_text_label_consistent = self._filter_by_text_length(example)

        filter_results = [
            is_duration_within_limits,
            is_text_label_consistent,
        ]

        if NUM_WORDS_FILTER in self.optional_filters:
            is_num_words_ok = self._filter_by_num_words(
                example,
                min_num_words=partition_spec.get("min_num_words", None),
                max_num_words=partition_spec.get("max_num_words", None),
            )
            filter_results.append(is_num_words_ok)

        if CTC_ALIGNER_FILTER in self.optional_filters:
            is_ctc_alignment_ok = self._filter_by_ctc_alignment_failure(
                example
            )
            if not is_ctc_alignment_ok:
                logger.warning(
                    f"CTC alignment failed for example "
                    f"{example[self.text_keys[0]]}. Removing it."
                )
            filter_results.append(is_ctc_alignment_ok)

        return all(filter_results)

    def _filter_by_duration(
        self,
        example: Dict[str, Any],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> bool:
        return all(
            self._filter_by_duration_for_keys(
                sequence_key=sequence_key,
                duration_key=duration_key,
                example=example,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            for sequence_key, duration_key in zip(
                self.sequence_keys, self.duration_keys
            )
        )

    def _filter_by_duration_for_keys(
        self,
        sequence_key: str,
        duration_key: str,
        example: Dict[str, Any],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> bool:
        if example[sequence_key] is None:
            return True

        if min_duration is None and max_duration is None:
            return True

        min_duration = min_duration or 0
        max_duration = max_duration or float("inf")

        duration = example.get(duration_key, None)
        if duration is None:
            duration = self._get_duration(example, sequence_key)

        return min_duration <= duration <= max_duration

    def _get_duration(
        self, example: Dict[str, Any], sequence_key: str
    ) -> float:
        return self.duration_getters[sequence_key](example)

    def _filter_by_text_length(self, example: Dict[str, Any]) -> bool:
        if self.token_rate_validation_triplets is None:
            return True

        return all(
            self._filter_by_text_length_for_keys(
                duration_key, text_key, max_tokens_per_minute, example
            )
            for duration_key, text_key, max_tokens_per_minute in self.token_rate_validation_triplets
        )

    def _filter_by_text_length_for_keys(
        self,
        duration_key: str,
        text_key: str,
        max_tokens_per_minute: int,
        example: Dict[str, Any],
    ) -> bool:
        if duration_key not in example:
            return True

        duration_in_seconds = example[duration_key]
        duration_in_minutes = duration_in_seconds / 60
        max_allowed_tokens = duration_in_minutes * max_tokens_per_minute

        def is_text_below_max_tokens(text_key):
            return len(example[text_key].split()) <= max_allowed_tokens

        return is_text_below_max_tokens(text_key)

    def _filter_by_num_words(
        self,
        example: Dict[str, Any],
        min_num_words: Optional[int] = None,
        max_num_words: Optional[int] = None,
    ) -> bool:
        return all(
            self._filter_by_num_words_for_keys(
                text_key=text_key,
                example=example,
                min_num_words=min_num_words,
                max_num_words=max_num_words,
            )
            for text_key in self.text_keys
        )

    def _filter_by_num_words_for_keys(
        self,
        text_key: str,
        example: Dict[str, Any],
        min_num_words: Optional[int] = None,
        max_num_words: Optional[int] = None,
    ) -> bool:
        if min_num_words is None and max_num_words is None:
            return True
        if text_key not in example:
            raise ValueError(f"Text key '{text_key}' not found in example")

        num_words = len(example[text_key].split())
        if min_num_words is not None and num_words < min_num_words:
            return False
        if max_num_words is not None and num_words > max_num_words:
            return False
        return True

    def _filter_by_ctc_alignment_failure(
        self, example: Dict[str, Any]
    ) -> bool:
        return all(
            self._filter_by_ctc_alignment_failure_for_keys(
                duration_key=duration_key,
                sequence_key=sequence_key,
                text_key=text_key,
                example=example,
            )
            for duration_key, text_key, sequence_key in zip(
                self.duration_keys, self.text_keys, self.sequence_keys
            )
        )

    def _filter_by_ctc_alignment_failure_for_keys(
        self,
        duration_key: str,
        text_key: str,
        sequence_key: str,
        example: Dict[str, Any],
    ) -> bool:
        import re

        import inflect
        import numpy as np

        lec = inflect.engine()

        def process_text(text: str):
            text = re.sub(
                r"\d+(\.\d+)?",
                lambda x: lec.number_to_words(x.group()),
                text.lower(),
            )
            text = re.sub(r"[^a-z\s]", "", text)
            return text.split()

        frame_rate = 50  # CTC alignment is done at 50 frames per second

        # get the total duration of the audio from the audio encoder output
        total_duration = example.get(duration_key, None)
        if total_duration is None:
            total_duration = self._get_duration(example, sequence_key)

        total_frames = int(np.ceil(total_duration * frame_rate))

        transcript = process_text(example[text_key])
        chars = len([c for word in transcript for c in word])
        return chars <= total_frames * 0.50


def get_audio_duration(
    example: Dict[str, Any], audio_key: str = "audio"
) -> float:
    if "duration" in example:
        return example["duration"]

    num_samples = len(example[audio_key]["array"])
    sampling_rate = example[audio_key]["sampling_rate"]
    return num_samples / sampling_rate


class CustomAudioDataset(CustomTemporalDataset):
    sequence_keys = ["audio"]
    text_keys = ["transcription"]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        target_sr = config.additional_args.get("sampling_rate", None)
        self.columns_to_cast = {
            "audio": Audio(sampling_rate=target_sr, mono=True, decode=True),
            "audio_output": Audio(
                sampling_rate=target_sr, mono=True, decode=True
            ),
            "audio_input": Audio(
                sampling_rate=target_sr, mono=True, decode=True
            ),
            "audio_condition": Audio(
                sampling_rate=target_sr, mono=True, decode=True
            ),
        }
        for key in self.sequence_keys:
            self.duration_getters.update({key: get_audio_duration})

        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )


def get_video_duration(example: Dict[str, Any]) -> float:
    if "duration" in example:
        return example["duration"]

    video_reader = VideoReader(BytesIO(example["video"]["bytes"]))
    duration = len(video_reader) / video_reader.get_avg_fps()
    del video_reader
    return duration


class CustomVideoDataset(CustomTemporalDataset):
    sequence_keys = ["video"]

    # NOTE: decoding is done in SpeechLmmDataset.__getitem__()
    columns_to_cast = {"video": Video(decode=False)}

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        for key in self.sequence_keys:
            self.duration_getters.update({key: get_video_duration})

        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )


class CustomAudioVisualDataset(CustomTemporalDataset):
    sequence_keys = ["audio", "video"]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        self.duration_getters.update(
            {
                "audio": get_audio_duration,
                # "video": get_video_duration,
            }
        )
        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
        )


class SpokenLanguageUnderstandingDataset(CustomAudioDataset):
    @abstractmethod
    def get_intents(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_slot_types(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_transcription_with_annotated_slots(
        self, transcription: str, example: Dict[str, Any]
    ) -> str:
        # NOTE(anferico): this function should return annotated
        # transcriptions in a standardized format
        raise NotImplementedError


class ConditioningMixin:
    def get_conditioning_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return None


class CaptionMixin:
    def get_caption_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        return None


class ExternalTranscriptionMixin:
    def get_external_transcription_for_example(
        self, example: Dict[str, Any]
    ) -> Optional[str]:
        return None


class FewShotsMixin(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.few_shots_split = kwargs.pop("few_shots_split", None)
        if self.few_shots_split is None:
            raise ValueError("`few_shots_split` must be provided.")
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_few_shot_examples(
        self, example: Dict[str, Any], n: int
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError
