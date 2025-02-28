import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

SLU_N_FEW_SHOT = 2

SOURCE_LANGUAGE_PLACEHOLDER = "<source-language>"
TARGET_LANGUAGE_PLACEHOLDER = "<target-language>"
TTS_SENTENCE_PLACEHOLDER = "<sentence>"
SQA_QUESTION_PLACEHOLDER = "<question>"

LANGUAGES_CODE_NAME = {
    "en": {
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
    },
    "es": {
        "en": "inglés",
        "es": "español",
        "fr": "francés",
        "de": "alemán",
        "it": "italiano",
    },
    "fr": {
        "en": "anglais",
        "es": "espagnol",
        "fr": "français",
        "de": "allemand",
        "it": "italien",
    },
    "de": {
        "en": "englisch",
        "es": "spanisch",
        "fr": "französisch",
        "de": "deutsch",
        "it": "italienisch",
    },
    "it": {
        "en": "inglese",
        "es": "spagnolo",
        "fr": "francese",
        "de": "tedesco",
        "it": "italiano",
    },
}

SLU_INTENT_STR = {
    "SpeechMassive": [
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
    ],
    "Slurp": [
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
    ],
}

SLU_SLOT_TYPE = {
    "SpeechMassive": [
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
    ],
    "Slurp": [
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
    ],
}

@dataclass
class BaseConfig:
    DATA: Dict
    OUTPUTS_TEXT_LIST: Optional[Dict] = None
    INPUTS_TEXT_LIST: Optional[Dict] = None


@dataclass
class CustomDatasetConfig:
    datapath: str
    task: str
    languages: List[str]
    partitions: Dict
    OUTPUTS_TEXT_LIST: Optional[Dict] = None
    INPUTS_TEXT_LIST: Optional[Dict] = None
    # any additional parameter contributing to the dataset's fingerprint
    additional_args: Optional[Dict] = None


class EnvVarSafeLoader(yaml.SafeLoader):
    """
    A YAML SafeLoader to process environment variables in YAML.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self.add_implicit_resolver(
            "!env_variable", re.compile(r"\$\{[^}]+\}"), None
        )
        self.add_constructor("!env_variable", type(self).env_constructor)

    @staticmethod
    def env_constructor(loader, node):
        """
        Constructor that replaces ${VAR_NAME} with the value of the VAR_NAME environment variable.
        """
        value = loader.construct_scalar(node)
        pattern = re.compile(r"\$\{([^}]+)\}")
        match = pattern.findall(value)
        if match:
            for var_name in match:
                env_value = os.getenv(var_name)
                if env_value is not None:
                    value = value.replace(f"${{{var_name}}}", env_value)
        return value


def safe_load_with_env_vars(stream):
    """
    Load a YAML file with environment variables.
    """
    return yaml.load(stream, EnvVarSafeLoader)


class DatasetsWrapperConfig:
    def __init__(
        self,
        config_path: str = None,
    ):
        self.config_path = config_path

    def from_yml(self):
        with open(self.config_path, "r") as f:
            config = safe_load_with_env_vars(f)
        for key in ["OUTPUTS_TEXT_LIST", "INPUTS_TEXT_LIST"]:
            if key in config and config[key] is not None:
                with open(f"{config[key]}", "r") as f:
                    config[key] = json.load(f)
            else:
                config[key] = None
        return BaseConfig(**config)
