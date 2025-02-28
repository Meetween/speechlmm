from speechlmm.dataset.custom_dataset.alpaca import AlpacaDataset
from speechlmm.dataset.custom_dataset.ami import AmiDataset
from speechlmm.dataset.custom_dataset.blizzard2013 import (
    BlizzardChallenge2013Dataset,
)
from speechlmm.dataset.custom_dataset.common_voice import CommonVoiceDataset
from speechlmm.dataset.custom_dataset.covost2 import Covost2Dataset
from speechlmm.dataset.custom_dataset.cvss import CvssDataset
from speechlmm.dataset.custom_dataset.daps_flores import DapsFloresDataset
from speechlmm.dataset.custom_dataset.databricks_dolly_15k import (
    DatabricksDolly15kDataset,
)
from speechlmm.dataset.custom_dataset.europarl import EuroparlDataset
from speechlmm.dataset.custom_dataset.everything_instruct_multilingual import (
    EverythingInstructMultilingualDataset,
)
from speechlmm.dataset.custom_dataset.grade_school_math_instructions import (
    GradeSchoolMathInstructionsDataset,
)
from speechlmm.dataset.custom_dataset.helpful_instructions import (
    HelpfulInstructionsDataset,
)
from speechlmm.dataset.custom_dataset.icsi import IcsiDataset
from speechlmm.dataset.custom_dataset.librilight import LibriLightDataset
from speechlmm.dataset.custom_dataset.libritts import LibriTtsDataset
from speechlmm.dataset.custom_dataset.lrs2 import LRS2VideoOnlyDataset
from speechlmm.dataset.custom_dataset.mls import MlsDataset
from speechlmm.dataset.custom_dataset.mustc import MustcDataset
from speechlmm.dataset.custom_dataset.nllb_paracrawl import NllbParacrawlDataset
from speechlmm.dataset.custom_dataset.oasst import OasstDataset
from speechlmm.dataset.custom_dataset.slurp import SlurpDataset
from speechlmm.dataset.custom_dataset.speech_massive import SpeechMassiveDataset
from speechlmm.dataset.custom_dataset.spoken_squad import SpokenSquadDataset
from speechlmm.dataset.custom_dataset.wikimedia import WikimediaDataset

DATASET_MAPPING = {
    "CoVoST": Covost2Dataset,
    "MuSTC": MustcDataset,
    "LibriTTS": LibriTtsDataset,
    "SpeechMassive": SpeechMassiveDataset,
    "Slurp": SlurpDataset,
    "Blizzard2013": BlizzardChallenge2013Dataset,
    "MLS": MlsDataset,
    "CVSS": CvssDataset,
    "Alpaca": AlpacaDataset,
    "LibriLight": LibriLightDataset,
    "CommonVoice": CommonVoiceDataset,
    "NllbParacrawl": NllbParacrawlDataset,
    "DapsFlores": DapsFloresDataset,
    "LRS2VideoOnly": LRS2VideoOnlyDataset,
    "Wikimedia": WikimediaDataset,
    "HelpfulInstructions": HelpfulInstructionsDataset,
    "GradeSchoolMathInstructions": GradeSchoolMathInstructionsDataset,
    "DatabricksDolly15k": DatabricksDolly15kDataset,
    "EverythingInstructMultilingual": EverythingInstructMultilingualDataset,
    "Oasst": OasstDataset,
    "AMI": AmiDataset,
    "ICSI": IcsiDataset,
    "Europarl": EuroparlDataset,
    "SpokenSQuAD": SpokenSquadDataset,
}
