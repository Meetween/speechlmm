import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import concatenate_datasets

from speechlmm.dataset.config import CustomDatasetConfig
from speechlmm.dataset.custom_dataset.base import (
    FewShotsMixin,
    SpokenLanguageUnderstandingDataset,
)
from speechlmm.dataset.custom_dataset.preparers import (
    SpeechRecognitionPreparer,
    SpokenLanguageUnderstandingPreparer,
)

logger = logging.getLogger(__name__)


# TODO(anferico): implement this
class SpeechMassiveDataset(FewShotsMixin, SpokenLanguageUnderstandingDataset):
    name = "SpeechMassive"
    codename = "speechmassive"
    splits = ["train", "eval", "test", "few_shots"]

    sequence_keys = ["audio"]
    text_keys = ["utt"]

    token_rate_validation_triplets = [("duration", "utt", 700)]

    def __init__(
        self,
        config: CustomDatasetConfig,
        cache_final_datasets: bool = False,
        rebuild_cache: bool = False,
        num_proc_for_preprocessing: Optional[int] = None,
    ):
        remap_keys = {"utt": "transcription"}
        remap_keys_slu = {"intent_str": "intent", **remap_keys}
        self.preparers = {
            "ASR": SpeechRecognitionPreparer(remap_keys=remap_keys),
            "SLU": SpokenLanguageUnderstandingPreparer(
                remap_keys=remap_keys_slu,
                dataset=self,
                do_slot_filling=True,
                num_few_shots=config.additional_args.get("num_few_shots", 0),
            ),
            "SLU_INTENT_ONLY": SpokenLanguageUnderstandingPreparer(
                remap_keys=remap_keys_slu,
                dataset=self,
                do_slot_filling=False,
                num_few_shots=config.additional_args.get("num_few_shots", 0),
            ),
        }

        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
            few_shots_split="few_shots",
        )
        few_shots_seed = config.additional_args.get("few_shots_seed", 42)
        self.few_shots_rng = np.random.default_rng(seed=few_shots_seed)

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        def add_locale(language):
            if language == "de":
                return "de-DE"
            elif language == "fr":
                return "fr-FR"
            elif language == "es":
                return "es-ES"
            else:
                raise ValueError(f"Unsupported language: {language}")

        dataset_path = Path(
            dataset_dir,
            partition_name,
            f"speechmassive.{partition_name}.{add_locale(source_language)}.parquet",
        )
        return str(dataset_path)

    def get_intents(self) -> List[str]:
        return [
            "alarm_query",
            "alarm_remove",
            "alarm_set",
            "audio_volume_down",
            "audio_volume_mute",
            "audio_volume_other",
            "audio_volume_up",
            "calendar_query",
            "calendar_remove",
            "calendar_set",
            "cooking_query",
            "cooking_recipe",
            "datetime_convert",
            "datetime_query",
            "email_addcontact",
            "email_query",
            "email_querycontact",
            "email_sendemail",
            "general_greet",
            "general_joke",
            "general_quirky",
            "iot_cleaning",
            "iot_coffee",
            "iot_hue_lightchange",
            "iot_hue_lightdim",
            "iot_hue_lightoff",
            "iot_hue_lighton",
            "iot_hue_lightup",
            "iot_wemo_off",
            "iot_wemo_on",
            "lists_createoradd",
            "lists_query",
            "lists_remove",
            "music_dislikeness",
            "music_likeness",
            "music_query",
            "music_settings",
            "news_query",
            "play_audiobook",
            "play_game",
            "play_music",
            "play_podcasts",
            "play_radio",
            "qa_currency",
            "qa_definition",
            "qa_factoid",
            "qa_maths",
            "qa_stock",
            "recommendation_events",
            "recommendation_locations",
            "recommendation_movies",
            "social_post",
            "social_query",
            "takeaway_order",
            "takeaway_query",
            "transport_query",
            "transport_taxi",
            "transport_ticket",
            "transport_traffic",
            "weather_query",
        ]

    def get_slot_types(self) -> List[str]:
        return [
            "transport_agency",
            "movie_type",
            "drink_type",
            "coffee_type",
            "meal_type",
            "food_type",
            "person",
            "transport_name",
            "song_name",
            "email_folder",
            "transport_type",
            "definition_word",
            "movie_name",
            "media_type",
            "music_album",
            "device_type",
            "audiobook_name",
            "business_name",
            "personal_info",
            "joke_type",
            "change_amount",
            "game_name",
            "place_name",
            "music_genre",
            "news_topic",
            "weather_descriptor",
            "time_zone",
            "event_name",
            "artist_name",
            "alarm_type",
            "time",
            "currency_name",
            "color_type",
            "order_type",
            "general_frequency",
            "house_place",
            "radio_name",
            "business_type",
            "relation",
            "cooking_type",
            "podcast_name",
            "game_type",
            "ingredient",
            "transport_descriptor",
            "player_setting",
            "audiobook_author",
            "playlist_name",
            "sport_type",
            "music_descriptor",
            "date",
            "list_name",
            "podcast_descriptor",
            "app_name",
            "timeofday",
            "email_address",
        ]

    def get_transcription_with_annotated_slots(
        self, transcription: str, example: Dict[str, Any]
    ) -> str:
        # NOTE(anferico): SpeechMassive already has annotated slots in
        # the correct format
        return example["annot_utt"]

    # TODO(anferico): factorize in a class that inherits from `FewShotsMixin`
    def get_few_shot_examples(
        self, example: Dict[str, Any], n: int = 2
    ) -> List[Dict[str, Any]]:
        # NOTE: in order to avoid sampling `example` itself, we
        # sample n+1 examples and then discard any sample (≤ 1) that is
        # equal to `example`
        few_shots_dataset = getattr(
            self, f"{self.few_shots_split}_dataset", None
        )
        if few_shots_dataset is None:
            raise ValueError(
                f"{self.__class__.__name__}.{self.few_shots_split}_dataset "
                f"must exist and be not None."
            )
        few_shots = few_shots_dataset.shuffle(
            generator=self.few_shots_rng
        ).select(range(n + 1))
        few_shots = few_shots.filter(lambda ex: ex != example)
        return few_shots.select(range(n))
