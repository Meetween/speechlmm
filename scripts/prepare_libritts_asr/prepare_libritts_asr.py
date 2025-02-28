import argparse
import json
import os
import random

import librosa

# import from constants.py file in the same dir of current file
from constants import FINE_TUNING_ANSWERS_LIST, PRETRAINING_QUESTIONS_LIST


def download_libritts(splits: list, libritts_dir: str):
    """
    Download LibriTTS dataset from the official source
    """

    if not os.path.exists(libritts_dir):
        os.makedirs(libritts_dir)
    url = "http://www.openslr.org/resources/60/"
    for split in splits:
        # if os.path.exists(f"{libritts_dir}/{split}.tar.gz"):
        #     print(f"{split}.tar.gz already exists, skipping download")
        #     continue
        os.system(f"wget {url}/{split}.tar.gz -P {libritts_dir}")
        os.system(f"tar -xzf {libritts_dir}/{split}.tar.gz -C {libritts_dir}")
        os.system(f"rm {libritts_dir}/{split}.tar.gz")
    print("LibriTTS dataset downloaded and extracted successfully")


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--libritts_dir", type=str)
    args.add_argument("--output_dir", type=str, default=None)
    args.add_argument("--download_libritts", action="store_true")
    args.add_argument("--pretraining", action="store_true")
    args.add_argument("--fine_tuning", action="store_true")

    args = args.parse_args()
    if args.pretraining and args.fine_tuning:
        raise ValueError(
            "Cannot perform both pretraining and fine-tuning at the same time"
        )
    if not args.pretraining and not args.fine_tuning:
        raise ValueError("Please specify either pretraining or fine-tuning")
    if args.pretraining:
        print("Pretraining mode selected")
    if args.fine_tuning:
        print("Fine-tuning mode selected")
    return args


def make_json(
    libritts_full_path: str,
    available_training_splits: list,
    out_file: str,
    finetuning: bool = False,
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
                        random_index = random.randint(
                            0, len(PRETRAINING_QUESTIONS_LIST) - 1
                        )
                        question = PRETRAINING_QUESTIONS_LIST[random_index]
                        answer = random.choice(
                            FINE_TUNING_ANSWERS_LIST[random_index]
                        )
                        answer = (
                            answer + " " + "'" + audio_text + "'"
                            if finetuning
                            else audio_text
                        )
                        conversations = [
                            {
                                "from": "human",
                                "value": question + "<audio>\n",
                            },
                            {"from": "gpt", "value": answer},
                        ]
                        audio_dict = {
                            "id": audio_id,
                            "audio": os.path.join(
                                split, speaker, book, audio_file
                            ),
                            "conversations": conversations,
                        }
                        # Check if the audio file length is between 1 and 20 seconds
                        duration = librosa.get_duration(
                            filename=os.path.join(
                                speaker_path, book, audio_file
                            )
                        )
                        if duration > 0.5 and duration < 25:
                            list_of_dicts.append(audio_dict)
                            accepted_files += 1
                        else:
                            print(
                                f"Audio file {audio_id} has a length of {duration} seconds, which is not between 0.5 and 25 seconds"
                            )
                            discarded_files += 1
        print()
        print(f"Accepted {accepted_files} audio files")
        print(f"Discarded {discarded_files} audio files due to length")
        random.shuffle(list_of_dicts)
        json.dump(list_of_dicts, f, indent=4)


if __name__ == "__main__":

    libritts_full = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    args = parse_args()

    if args.download_libritts:
        if args.libritts_dir is None:
            raise ValueError(
                "Please specify the LibriTTS directory to download the dataset"
            )
        download_libritts(libritts_full, args.libritts_dir)
    else:
        assert os.path.exists(
            args.libritts_dir
        ), f"LibriTTS directory not found at {args.libritts_dir}"

    if args.output_dir is None:
        args.output_dir = args.libritts_dir
    else:
        assert os.path.exists(
            args.output_dir
        ), f"Output directory not found at {args.output_dir}"

    libritts_full_path = os.path.join(args.libritts_dir, "LibriTTS")
    available_training_splits = [
        x
        for x in os.listdir(libritts_full_path)
        if "train" in x and x in libritts_full
    ]
    print(f"Available training splits: {available_training_splits}")

    if args.pretraining:
        out_file = os.path.join(args.output_dir, "libritts_pretraining.json")

    if args.fine_tuning:
        out_file = os.path.join(args.output_dir, "libritts_finetuning.json")

    make_json(
        libritts_full_path,
        available_training_splits,
        out_file,
        args.fine_tuning,
    )
