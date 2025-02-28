import os
import time

import h5py
import numpy as np
import torchaudio

# This script will convert the entire LibriTTS dataset into a single hdf5 file.
# it can be adapted to work with other datasets easily, but be careful since
# it assumes that a single split of the dataset can fit into memory


LIBRITTS_FOLDER = "playground/data/LLaVA-speech_Pretrain/LibriTTS"
possible_splits = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]


def make_hdf5_file(split_name):
    outputs = []
    wavs = []
    for root, dirs, files in os.walk(
        os.path.join(LIBRITTS_FOLDER, split_name)
    ):
        for file in files:
            if file.endswith(".wav"):
                wavs.append(os.path.join(root, file))
    print(f"Processing {split_name} split with {len(wavs)} files")
    start = time.time()
    for wav in wavs:
        waveform, sample_rate = torchaudio.load(wav)
        wav_without_LIBRITTS_FOLDER = wav.replace(LIBRITTS_FOLDER + "/", "")
        text_file = wav.replace(".wav", ".normalized.txt")
        with open(text_file, "r") as f:
            transcription = f.read().strip()
        outputs.append(
            (wav_without_LIBRITTS_FOLDER, waveform, sample_rate, transcription)
        )

    with h5py.File(
        f"playground/data/LLaVA-speech_Pretrain/LibriTTS/{split_name}.hdf5",
        "w",
    ) as f:
        for audio_name, waveform, sample_ratem, transcription in outputs:
            dset = f.create_dataset(audio_name, data=waveform.numpy())
            dset.attrs["sample_rate"] = sample_rate
            dset.attrs["transcription"] = transcription
    end = time.time()
    print(f"Finished processing {split_name} split in {end-start:.2f} seconds")


for split in possible_splits:
    make_hdf5_file(split)
