import argparse
import json
import os
import random
import re

import librosa
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PRETRAINING_QUESTIONS_LIST = [
    "Can you transcribe the audio content?",
    "What is being said in the audio? Is ASR possible?",
    "Could you perform automatic speech recognition on this audio?",
    "What are the words spoken in this audio clip?",
    "Is it possible to convert this audio to text?",
    "Can you decipher the audio's speech content?",
    "What does the sound clip articulate? Can ASR be applied?",
    "Can you extract text from this audio?",
    "What is the verbal content of this audio?",
    "Could you provide a transcription for this audio?",
    "Is converting this audio to text feasible?",
    "Can the spoken parts of this audio be identified?",
    "What speech is contained within this audio file?",
    "Can you decode the audio into text?",
    "What does this audio recording express?",
    "Could you analyze the audio and provide the text?",
    "Is it possible to do speech-to-text on this audio?",
    "Can the dialogue in the audio be transcribed?",
    "What are the uttered words in this audio?",
    "Could you perform speech recognition on this audio?",
]


def download_libritts(splits: list, libritts_dir: str):
    """
    Download LibriTTS dataset from the official source
    """
    if not os.path.exists(libritts_dir):
        os.makedirs(libritts_dir)
    url = "http://www.openslr.org/resources/60/"
    for split in splits:
        split_path = os.path.join(libritts_dir, split)
        if not os.path.exists(split_path):
            os.system(f"wget {url}/{split}.tar.gz -P {libritts_dir}")
            os.system(
                f"tar -xzf {libritts_dir}/{split}.tar.gz -C {libritts_dir}"
            )
            os.system(f"rm {libritts_dir}/{split}.tar.gz")
    print("LibriTTS dataset downloaded and extracted successfully")


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--store_dir", type=str)
    args.add_argument("--output_dir", type=str, default=None)

    args = args.parse_args()

    return args


def make_json(
    libritts_full_path: str, available_training_splits: list, out_file: str
):
    """
    Create a JSON file for pretraining using the LibriTTS dataset
    """

    # The file will actually contain a list of dictionaries in json format.
    # for each audio file, we will have a dictionary with the following keys:
    # id, audio, convesations. Conversations will be a list of dictionaries with the following keys
    # from, value. For pretraining, it will always be two: one where the "from" is
    # the "human" and the other where the "from" is "gpt".
    list_of_dicts = []
    discarded_files = 0
    accepted_files = 0
    with open(out_file, "w") as f:
        for split in available_training_splits:
            split_path = os.path.join(libritts_full_path, split)
            speakers = [
                x
                for x in os.listdir(os.path.join(split_path))
                if os.path.isdir(os.path.join(split_path, x))
            ]
            for speaker in speakers:
                speaker_path = os.path.join(split_path, speaker)
                books = [x for x in os.listdir(speaker_path)]
                for book in books:
                    book_path = os.path.join(speaker_path, book)
                    audio_files = [
                        x for x in os.listdir(book_path) if x.endswith(".wav")
                    ]
                    for audio_file in audio_files:
                        audio_id = audio_file.split(".")[0]
                        audio_text_file = audio_file.replace(
                            ".wav", ".normalized.txt"
                        )
                        with open(
                            os.path.join(speaker_path, book, audio_text_file),
                            "r",
                        ) as t:
                            audio_text = t.read().strip()

                        audio_dict = {
                            "id": audio_id,
                            "audio": os.path.join(
                                split, speaker, book, audio_file
                            ),
                            "transcription": audio_text,
                        }
                        # Check if the audio file length is between 1 and 20 seconds
                        duration = librosa.get_duration(
                            filename=os.path.join(
                                speaker_path, book, audio_file
                            )
                        )
                        if duration > 2 and duration < 25:
                            list_of_dicts.append(audio_dict)
                            accepted_files += 1
                        else:
                            # print(f"Audio file {audio_id} has a length of {duration} seconds, which is not between 0.5 and 25 seconds")
                            discarded_files += 1
        print()
        print(f"Accepted {accepted_files} audio files")
        print(f"Discarded {discarded_files} audio files due to length")
        random.shuffle(list_of_dicts)
        json.dump(list_of_dicts, f, indent=4)


if __name__ == "__main__":

    libritts_train = [
        "train-clean-100",
        "train-clean-360",
    ]  # "train-other-500"]
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = args.store_dir
    else:
        assert os.path.exists(
            args.output_dir
        ), f"Output directory not found at {args.output_dir}"

    libritts_train_path = os.path.join(args.store_dir, "LibriTTS")
    not_available_training_splits = [
        x for x in libritts_train if x not in os.listdir(libritts_train_path)
    ]
    print(not_available_training_splits)

    if args.store_dir is None:
        raise ValueError(
            "Please specify the store directory to download and store the dataset"
        )
    download_libritts(not_available_training_splits, args.store_dir)

    out_file = os.path.join(args.output_dir, "libritts_sqa.json")

    make_json(libritts_train_path, libritts_train, out_file)
