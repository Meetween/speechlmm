# This is an experimental script that
# will extend the pretraining transcription dataset with translation to italian
# and generic instructions not related to transcription, taken from alpaca
# https://github.com/gururise/AlpacaDataCleaned


import argparse
import copy
import json
import os
import random

from tqdm import tqdm
from transformers import pipeline

TRANSLATION_REQUESTS = [
    "Translate audio to italian, please.",
    "Need the audio translated into italian.",
    "Italian translation, please.",
    "Can you provide an Italian translation of the audio?",
    "Audio to Italian?",
    "I'm looking for a translation of the audio in Italian.",
    "Translate to Italian?",
    "I'd like to request an Italian translation of the audio, please.",
    "Italian translation needed.",
    "Can you translate the audio into italian for me?",
    "Translate to Italian.",
    "I'm interested in getting the audio translated into Italian.",
    "Can you turn the audio into italian, please?",
    "Is there any chance you could translate the audio to Italian?",
    "Could you kindly translate the audio into Italian?",
    "Traduci l'audio in italiano",
    "Ho bisogno dell'audio tradotto in italiano.",
    "Traduzione in italiano, per favore.",
    "Puoi fornire una traduzione in italiano dell'audio?",
    "Audio in italiano?",
    "Sto cercando una traduzione dell'audio in italiano.",
    "Traduci in italiano",
    "Vorrei richiedere una traduzione in italiano dell'audio, per favore.",
    "Traduzione in italiano necessaria.",
    "Puoi tradurre l'audio in italiano per me?",
    "Traduci in italiano.",
    "Sono interessato a ottenere l'audio tradotto in italiano.",
    "Puoi trasformare l'audio in italiano, per favore?",
    "C'è qualche possibilità che tu possa tradurre l'audio in italiano?",
    "Potresti gentilmente tradurre l'audio in italiano?",
]


def load_translation_model():
    # put the pipe in gpu if available
    pipe = pipeline(
        "translation", model="Helsinki-NLP/opus-mt-tc-big-en-it", device=0
    )
    return pipe


def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Extend the pretraining transcription dataset with translation to italian and generic instructions not related to transcription, taken from alpaca"
    )
    parser.add_argument(
        "--alpaca_path",
        type=str,
        default="alpaca_data_cleaned.json",
        help="Path to the alpaca dataset",
    )
    parser.add_argument(
        "--transcription_path",
        type=str,
        default="playground/data/LLaVA-speech_Pretrain/libritts_pretraining.json",
        help="Path to the transcription dataset",
    )
    return parser.parse_args()


def translate_transcription_dataset(transcription_dataset, pipe):

    pipe = pipeline(
        "translation", model="Helsinki-NLP/opus-mt-tc-big-en-it", device=0
    )
    translated_dataset = []
    for conversation in tqdm(transcription_dataset):
        translated_conversation = copy.deepcopy(conversation)
        translated_conversation["conversations"][0]["value"] = (
            random.choice(TRANSLATION_REQUESTS) + "<audio>\n"
        )
        translated_conversation["conversations"][1]["value"] = pipe(
            conversation["conversations"][1]["value"]
        )[0]["translation_text"]
        translated_dataset.append(translated_conversation)

    return translated_dataset


def build_instruction_dataset(transcription_dataset, alpaca_dataset):
    instruction_dataset = []
    for transcription_conversation in tqdm(transcription_dataset):
        new_conversation = copy.deepcopy(transcription_conversation)
        instruction_conversation = random.choice(alpaca_dataset)
        new_conversation["conversations"][0]["value"] = (
            instruction_conversation["instruction"] + "<audio>\n"
        )
        new_conversation["conversations"][1]["value"] = (
            instruction_conversation["output"]
        )
        instruction_dataset.append(new_conversation)

    return instruction_dataset


def main():
    args = arg_parser()
    alpaca_dataset = load_dataset(args.alpaca_path)
    transcription_dataset = load_dataset(args.transcription_path)
    pipe = load_translation_model()
    translated_dataset = translate_transcription_dataset(
        transcription_dataset, pipe
    )
    instruction_dataset = build_instruction_dataset(
        transcription_dataset, alpaca_dataset
    )

    merged_dataset = (
        transcription_dataset + translated_dataset + instruction_dataset
    )
    shuffled_dataset = random.sample(merged_dataset, len(merged_dataset))
    # save the dataset
    with open(
        "playground/data/LLaVA-speech_Pretrain/extended_pretraining_dataset.json",
        "w",
    ) as f:
        json.dump(shuffled_dataset, f, indent=4)


if __name__ == "__main__":
    main()
