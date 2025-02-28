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


class SlurpDataset(FewShotsMixin, SpokenLanguageUnderstandingDataset):
    name = "SLURP"
    codename = "slurp"
    splits = ["train", "eval", "test"]

    sequence_keys = ["audio"]
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
            "SLU": SpokenLanguageUnderstandingPreparer(
                remap_keys=remap_keys,
                dataset=self,
                do_slot_filling=True,
                num_few_shots=config.additional_args.get("num_few_shots", 0),
            ),
            "SLU_INTENT_ONLY": SpokenLanguageUnderstandingPreparer(
                remap_keys=remap_keys,
                dataset=self,
                do_slot_filling=False,
                num_few_shots=config.additional_args.get("num_few_shots", 0),
            ),
        }

        # TODO(anferico): duplicated across this and SpeechMassive
        few_shots_split = config.additional_args.get(
            "few_shots_split", "train"
        )
        super().__init__(
            config,
            cache_final_datasets=cache_final_datasets,
            rebuild_cache=rebuild_cache,
            num_proc_for_preprocessing=num_proc_for_preprocessing,
            few_shots_split=few_shots_split,
        )
        few_shots_seed = config.additional_args.get("few_shots_seed", 42)
        self.few_shots_rng = np.random.default_rng(seed=few_shots_seed)

        if self.config.languages != ["en"]:
            logger.warning(
                f"Only English language is supported for {self.name}. Other languages will be ignored."
            )

    def _get_dataset_path(
        self,
        dataset_dir: str,
        source_language: str,
        target_language: str,
        partition_name: str,
        partition_spec: dict,
    ) -> str:
        dataset_path = Path(dataset_dir, f"slurp.{partition_name}.parquet")
        return str(dataset_path)

    def get_intents(self) -> List[str]:
        return [
            "addcontact",
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
            "cleaning",
            "coffee",
            "convert",
            "cooking_query",
            "cooking_recipe",
            "createoradd",
            "currency",
            "datetime_convert",
            "datetime_query",
            "definition",
            "email_addcontact",
            "email_query",
            "email_querycontact",
            "email_sendemail",
            "events",
            "factoid",
            "game",
            "general_greet",
            "general_joke",
            "general_quirky",
            "greet",
            "hue_lightdim",
            "hue_lightoff",
            "hue_lightup",
            "iot_cleaning",
            "iot_coffee",
            "iot_hue_lightchange",
            "iot_hue_lightdim",
            "iot_hue_lightoff",
            "iot_hue_lighton",
            "iot_hue_lightup",
            "iot_wemo_off",
            "iot_wemo_on",
            "joke",
            "lists_createoradd",
            "lists_query",
            "lists_remove",
            "music",
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
            "podcasts",
            "post",
            "qa_currency",
            "qa_definition",
            "qa_factoid",
            "qa_maths",
            "qa_stock",
            "query",
            "querycontact",
            "quirky",
            "radio",
            "recommendation_events",
            "recommendation_locations",
            "recommendation_movies",
            "remove",
            "sendemail",
            "set",
            "settings",
            "social_post",
            "social_query",
            "takeaway_order",
            "takeaway_query",
            "ticket",
            "traffic",
            "transport_query",
            "transport_taxi",
            "transport_ticket",
            "transport_traffic",
            "volume_other",
            "weather_query",
            "wemo_off",
            "wemo_on",
        ]

    def get_slot_types(self) -> List[str]:
        return [
            "song_name",
            "definition_word",
            "joke_type",
            "meal_type",
            "transport_name",
            "ingredient",
            "person",
            "time_zone",
            "alarm_type",
            "transport_descriptor",
            "game_name",
            "drink_type",
            "food_type",
            "music_genre",
            "order_type",
            "music_album",
            "media_type",
            "movie_type",
            "playlist_name",
            "device_type",
            "cooking_type",
            "email_address",
            "app_name",
            "change_amount",
            "list_name",
            "coffee_type",
            "personal_info",
            "artist_name",
            "radio_name",
            "weather_descriptor",
            "time",
            "audiobook_name",
            "general_frequency",
            "transport_type",
            "business_type",
            "house_place",
            "email_folder",
            "audiobook_author",
            "place_name",
            "relation",
            "sport_type",
            "game_type",
            "transport_agency",
            "color_type",
            "podcast_name",
            "timeofday",
            "date",
            "business_name",
            "player_setting",
            "podcast_descriptor",
            "currency_name",
            "movie_name",
            "music_descriptor",
            "event_name",
            "news_topic",
        ]

    def get_transcription_with_annotated_slots(
        self, transcription: str, example: Dict[str, Any]
    ) -> str:
        """
        Example:
        example["slots:"] = [ "email_recipient=John Doe", "time=4 pm" ]
        transcription = "Send an email to John Doe at 4 pm"
        returns "Send an email to [email_recipient : John Doe] at [time : 4 pm]"
        """

        # NOTE: the transcription *might* contain different slot types
        # with the same value. Assuming such slots are listed in
        # example["slots:"] in the same order as they appear in the
        # transcription, we replace the nth occurence of each slot value
        # with the nth [slot_type : slot_value] pair
        def replace_nth_occurrence(string, old, new, n):
            if n == 0:
                return string
            parts = string.split(old, n)
            if len(parts) <= n:
                return string
            return old.join(parts[:-1]) + new + parts[-1]

        # If there are no slots, we return the transcription as is
        transcription_with_tagged_slots = transcription

        count_slots_values = defaultdict(int)
        for slot in example["slots:"]:  # yes, with the colon...
            slot_type, slot_value = slot.split("=")
            slot_value = slot_value.lower()
            if slot_value not in transcription:
                # some slot values have typos or additional whitespaces
                # fmt: off
                slot_value = (
                    slot_value
                    .replace(" 's", "'s")
                    .replace(" ,", ",")
                    .replace("grassmarket", "grass market")
                    .replace("rihana", "rihanna")
                    .replace("barack obaba", "barack obama")
                    .replace("mail i d", "mail id")
                )
                # fmt: on
                if slot_value not in transcription:
                    logger.warning(
                        f'Slot value `{slot_value}` not found in "{transcription}"'
                    )
                    continue

            count_slots_values[slot_value] += 1

            # replace count_slots_values[slot_value]th occurence of
            # slot_value with [slot_type : slot_value]
            transcription_with_tagged_slots = replace_nth_occurrence(
                string=transcription,
                old=slot_value,
                new=f"[{slot_type} : {slot_value}]",
                n=count_slots_values[slot_value],
            )

        return transcription_with_tagged_slots

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
