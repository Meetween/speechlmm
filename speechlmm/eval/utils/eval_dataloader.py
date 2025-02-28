import ast
import json
import os
import random

import librosa
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset

from speechlmm.arguments import DataArguments
from speechlmm.constants import IGNORE_INDEX
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.dataset.speechlmm_dataset import process_video


def get_dataloader(dataset, batch_size=1, shuffle=True):
    def collate_fn(instances):
        batch = {}

        keys = [
            "audio",
            "gt",
            "task",
            "transcription",
            "source_language",
            "target_language",
            "sr",
            "video",
        ]

        for key in keys:
            batch[key] = []

        for instance in instances:
            for key in instance:
                # check if key is in instance
                if key not in keys:
                    continue
                if key == "audio":
                    has_audio = instance[key] is not None
                    sr = instance[key]["sampling_rate"] if has_audio else None
                    audio_array = torch.Tensor(instance[key]["array"])
                    batch[key].append(audio_array)
                    batch["sr"].append(sr)
                elif key == "video":
                    video = process_video(instance[key])
                    batch[key].append(video)
                else:
                    batch[key].append(instance[key])
            # extract gt from conversations: instance["conversations"][1]["value"]
            batch["gt"].append(instance["conversations"][1]["value"])

        return batch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )


class OldFormatLibriTTSLibriSpeechDataset(Dataset):
    """This function is still using the old LibriTTS and LibriSpeech dataset structure."""

    """New datasets will instead be loaded from parquet files."""

    """The outputs of all iterators should be tuples of the form (id, (audio_tensor:torch.Tensor, sample_rate:int))"""

    def __init__(self, data_dir, dataset_path):
        super().__init__()
        self.dataset_path = os.path.join(
            data_dir, "old_format_ASR_datasets", dataset_path
        )
        if "libritts" in dataset_path:
            all_audios = self.get_all_audios(self.dataset_path, "wav")
        elif "librispeech" in dataset_path:
            all_audios = self.get_all_audios(self.dataset_path, "flac")
        else:
            raise NotImplementedError(
                "Only LibriTTS and LibriSpeech datasets should be used with this function."
            )
        self.all_audios = all_audios
        assert (
            len(self.all_audios) > 0
        ), f"No audio files found in {self.dataset_path}, please check the path."

        # self.gt is initialized by dataset_name.jsonl found in the same directory as the dataset
        self.gt_dict = {}
        gt_path = os.path.join(
            data_dir,
            "old_format_ASR_datasets",
            dataset_path,
            f"{dataset_path}.jsonl",
        )
        with open(gt_path, "r") as f:
            for line in f:
                line = json.loads(line)
                self.gt_dict[line["id"]] = line["transcript"]

    @staticmethod
    def load_audio(audio_path):
        """
        Load audio from a local file path or URL using torchaudio.

        Parameters:
        - audiopath (str): The local path or URL of the audio file.

        Returns:
        - waveform (torch.Tensor): 1D tensor representing the audio waveform.
        - sample_rate (int): The sample rate of the audio.
        """
        try:
            # Load audio using torchaudio
            audio, sr = torchaudio.load(audio_path, normalize=True)
            return (audio, sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise e

    def get_all_audios(self, path, format):
        assert format in ["flac", "wav"]
        format_suffix = f".{format}"
        all_audios = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(format_suffix):
                    all_audios.append(
                        (
                            file.replace(format_suffix, ""),
                            os.path.join(root, file),
                        )
                    )
        return sorted(all_audios, key=lambda x: x[0])

    def __len__(self):
        return len(self.all_audios)

    def __getitem__(self, idx):
        audio_id, audio_path = self.all_audios[idx]
        audio, sr = self.load_audio(audio_path)
        audio = torch.Tensor(audio)
        gt = self.gt_dict[audio_id]
        task = "ASR"
        transcription = gt
        source_language = "en"
        target_language = None

        return {
            "audio": audio,
            "gt": gt,
            "task": task,
            "transcription": transcription,
            "source_language": source_language,
            "target_language": target_language,
            "sr": sr,
        }

    def get_dataloader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
            batch = {}
            keys = [
                "audio",
                "gt",
                "task",
                "transcription",
                "source_language",
                "target_language",
                "sr",
            ]
            for key in keys:
                batch[key] = []

            for instance in instances:
                for key in instance:
                    if key not in keys:
                        continue
                    batch[key].append(instance[key])

            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


class MustCEvaluationServerDataset(Dataset):
    def __init__(self, data_dir, direction):
        super().__init__()
        # direction can only be "en" if ASR, else
        # one of en-cs, en-de, en-es, en-fr, en-it, en-nl, en-pt, en-ro
        self.direction = direction
        # check that direction is valid
        assert direction in [
            "en",
            "en_cs",
            "en_de",
            "en_es",
            "en_fr",
            "en_it",
            "en_nl",
            "en_pt",
            "en_ro",
        ], f"Invalid split: {direction}"
        if direction == "en":
            self.dataset_path = os.path.join(
                data_dir, "evaluation_server_datasets", "ASR", "MUSTC", "en"
            )
            self.task = "ASR"
            self.source = "en"
            self.target = None
            self.yaml_path = os.path.join(self.dataset_path, "tst-COMMON.yaml")
            self.target_path = os.path.join(self.dataset_path, "tst-COMMON.en")
        else:
            self.dataset_path = os.path.join(
                data_dir,
                "evaluation_server_datasets",
                "ST",
                "MUSTC",
                direction.replace("_", "-"),
            )
            self.task = "ST"
            self.source = "en"
            self.target = direction.split("_")[1]
            self.yaml_path = os.path.join(self.dataset_path, "tst-COMMON.yaml")
            self.target_path = os.path.join(
                self.dataset_path,
                f"tst-COMMON.{direction.replace('_','-')}.{self.target}",
            )
            self.transcription_path = os.path.join(
                self.dataset_path,
                f"tst-COMMON.{direction.replace('_','-')}.en",
            )
        with open(self.target_path, "r") as f:
            self.target_lines = f.readlines()
        with open(self.yaml_path, "r") as f:
            self.yaml_lines = f.readlines()
            self.yaml_dict = []
            for line in self.yaml_lines:
                # load line as yaml
                self.yaml_dict.append(
                    yaml.load(line, Loader=yaml.SafeLoader)[0]
                )
        if self.task == "ST":
            with open(self.transcription_path, "r") as f:
                self.transcription_lines = f.readlines()

        assert len(self.target_lines) == len(
            self.yaml_dict
        ), "yaml and target files do not have the same length"
        self.wav_folder = os.path.join(self.dataset_path, "wav")

    def __len__(self):
        return len(self.target_lines)

    def __getitem__(self, idx):

        audio_dict = self.yaml_dict[idx]
        audio_path = os.path.join(self.wav_folder, audio_dict["wav"])
        offset = audio_dict["offset"]
        duration = audio_dict["duration"]
        file_path = os.path.join(self.wav_folder, audio_dict["wav"])
        y, sr = librosa.load(
            file_path, offset=offset, duration=duration, sr=None
        )
        if duration < 0.1:
            # pad the audio with zeros until it is at least 0.1 seconds long. This is to avoid errors in the model inference
            y = librosa.util.fix_length(y, size=int(0.1 * sr))
        audio = torch.tensor(y)
        task = self.task
        if self.task == "ASR":
            gt = self.target_lines[idx]
            transcription = gt
        else:
            gt = self.target_lines[idx]
            transcription = self.transcription_lines[idx]
        source_language = self.source
        target_language = self.target

        return {
            "audio": audio,
            "gt": gt,
            "task": task,
            "transcription": transcription,
            "source_language": source_language,
            "target_language": target_language,
            "sr": sr,
        }

    def get_dataloader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
            batch = {}
            keys = [
                "audio",
                "gt",
                "task",
                "transcription",
                "source_language",
                "target_language",
                "sr",
            ]
            for key in keys:
                batch[key] = []

            for instance in instances:
                for key in instance:
                    if key not in keys:
                        continue
                    batch[key].append(instance[key])

            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


class JamescalamYoutubeEvaluationDataset(Dataset):
    def __init__(self, data_dir, max_chunks=None):
        super().__init__()
        self.dataset_path = os.path.join(
            data_dir, "jamescalam-youtube-transcriptions"
        )
        transcriptions_path = os.path.join(
            self.dataset_path, "asr_jamescalam-youtube-transcriptions.json"
        )

        with open(transcriptions_path, "r") as f:
            self.transcriptions = json.load(f)
        for transcription in self.transcriptions:
            transcription["timestamps"] = ast.literal_eval(
                transcription["timestamps"]
            )  # safer than using eval

        # now get all audio ids
        audio_ids = set()
        for transcription in self.transcriptions:
            audio_ids.add(transcription["id"])
        audio_ids = sorted(list(audio_ids))
        # take first 2%
        audio_ids = audio_ids[: int(len(audio_ids) * 0.02)]

        self.transcriptions_dict = {}
        for audio_id in audio_ids:
            self.transcriptions_dict[audio_id] = []
        for transcription in self.transcriptions:
            if transcription["id"] in self.transcriptions_dict:
                self.transcriptions_dict[transcription["id"]].append(
                    transcription
                )
            else:
                continue

        if max_chunks is not None:
            self.max_chunks = max_chunks
        else:
            print(
                "WARNING: No max duration specified, using 2 chunks (max 60s each)"
            )
            self.max_chunks = 2

        # Get the list that can be used by the getitem function to get the audio files
        self.half_processed_audio_files = []

        for audio_id in audio_ids:
            all_transcriptions_for_audio_id = self.transcriptions_dict[
                audio_id
            ]
            # we want to split them into lists containing self.max_chunks transcriptions and put them in a list
            for i in range(
                0,
                len(all_transcriptions_for_audio_id) - self.max_chunks + 1,
                self.max_chunks,
            ):

                list_to_add = []
                for j in range(self.max_chunks):
                    list_to_add.append(all_transcriptions_for_audio_id[i + j])
                self.half_processed_audio_files.append(list_to_add)

    def __len__(self):
        return len(self.half_processed_audio_files)

    def __getitem__(self, idx):
        def __len__(self):
            return len(self.half_processed_audio_files)

    def _get_full_transcription(self, half_processed_audio):
        transcription_list = []
        for transcription in half_processed_audio:
            for split_transcription in transcription["timestamps"]:
                transcription_list.append(split_transcription["text"])
        return " ".join(transcription_list)

    def _get_full_audio(self, half_processed_audio):
        relative_audio_paths = []
        for transcription in half_processed_audio:
            relative_audio_paths.append(
                transcription["audio"].replace("./", "")
            )
        abs_paths = [
            os.path.join(self.dataset_path, path)
            for path in relative_audio_paths
        ]
        # load and concatenate the audio files

        audios = []
        for path in abs_paths:
            y, sr = librosa.load(path, sr=None)
            y = torch.tensor(y)
            audios.append(y)

        audio = torch.cat(audios)

        return (audio, sr)

    def __getitem__(self, idx):

        half_processed_audio = self.half_processed_audio_files[idx]
        audio, sr = self._get_full_audio(half_processed_audio)
        gt = self._get_full_transcription(half_processed_audio)

        task = "ASR"
        transcription = gt
        source_language = "en"
        target_language = None

        return {
            "audio": audio,
            "gt": gt,
            "task": task,
            "transcription": transcription,
            "source_language": source_language,
            "target_language": target_language,
            "sr": sr,
        }

    def get_dataloader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
            batch = {}
            keys = [
                "audio",
                "gt",
                "task",
                "transcription",
                "source_language",
                "target_language",
                "sr",
            ]
            for key in keys:
                batch[key] = []

            for instance in instances:
                for key in instance:
                    if key not in keys:
                        continue
                    batch[key].append(instance[key])

            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


class EuroParlST(Dataset):
    def __init__(
        self,
        data_dir,
        direction=None,
        split="test",
        max_duration=None,
        aim_for_min_duration=None,
        filter_for_data_beetween=None,
    ):
        super().__init__()
        self.dataset_path = os.path.join(data_dir, "europarl-ST", "v1.1")
        languages = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]
        source, target = direction.split("_")
        self.source = source
        self.target = target
        assert source in languages, f"Invalid source language: {source}"
        assert target in languages, f"Invalid target language: {target}"
        assert (
            source != target
        ), f"Source and target languages must be different"
        base_audio_path = os.path.join(self.dataset_path, source, "audios")
        base_texts_path = os.path.join(
            self.dataset_path, source, target, split
        )
        segments_lst = []
        with open(os.path.join(base_texts_path, "segments.lst"), "r") as f:
            for line in f:
                segments_lst.append(line.strip().split())
        segments_source = []
        with open(
            os.path.join(base_texts_path, f"segments.{source}"), "r"
        ) as f:
            segments_source = f.readlines()
        segments_target = []
        with open(
            os.path.join(base_texts_path, f"segments.{target}"), "r"
        ) as f:
            segments_target = f.readlines()

        list_of_data_dicts = []
        for i, segment in enumerate(segments_lst):
            audio_id = segment[0]
            audio_start = float(segment[1])
            audio_end = float(segment[2])
            duration = audio_end - audio_start
            audio_path = os.path.join(base_audio_path, f"{audio_id}.mp3")
            transcription = segments_source[i].strip()
            translation = segments_target[i].strip()
            list_of_data_dicts.append(
                {
                    "audio_paths": audio_path,
                    "audio_start": audio_start,
                    "audio_end": audio_end,
                    "duration": duration,
                    "transcription": transcription,
                    "translation": translation,
                }
            )
        self.data = list_of_data_dicts
        self.max_duration = max_duration
        if self.max_duration is not None:
            self.data = [
                d for d in self.data if d["duration"] <= self.max_duration
            ]
        # print all durations
        if aim_for_min_duration is not None:
            self._get_data_for_duration(aim_for_min_duration)

        if filter_for_data_beetween is not None:
            min = filter_for_data_beetween[0]
            max = filter_for_data_beetween[1]
            assert (
                min < max
            ), f"Invalid filter_for_data_beetween: {filter_for_data_beetween}"
            self.data = [
                d
                for d in self.data
                if d["duration"] >= min and d["duration"] <= max
            ]
            assert (
                len(self.data) > 0
            ), f"No data found for filter_for_data_beetween: {filter_for_data_beetween}"

    def _get_data_for_duration(self, aim_for_min_duration):
        # this is a leetcode-easy like exercise to join some of the previous dicts
        # We have to merge dict (merge the values, that are 1-element lists) to make sure that the duration is as close as possible to aim_for_min_duration
        # bear in mind that two audios can be merged ONLY if theu have the same audio_path
        new_data = []
        i = 0
        j = 1
        while i < len(self.data):
            indexes_to_merge = [i]
            if j == len(self.data):
                new_data.append(self.data[i])
                break
            total_duration = self.data[i]["duration"]
            while (
                j < len(self.data)
                and total_duration < aim_for_min_duration
                and self.data[i]["audio_paths"] == self.data[j]["audio_paths"]
            ):
                total_duration += self.data[j]["duration"]
                indexes_to_merge.append(j)
                j += 1
            # merge the dicts with indexes_to_merge
            new_dict = {
                "audio_paths": self.data[i]["audio_paths"],
                "audio_start": self.data[i]["audio_start"],
                "audio_end": self.data[indexes_to_merge[-1]]["audio_end"],
                "duration": self.data[indexes_to_merge[-1]]["audio_end"]
                - self.data[i]["audio_start"],
                "transcription": " ".join(
                    [
                        self.data[k]["transcription"].strip()
                        for k in indexes_to_merge
                    ]
                ),
                "translation": " ".join(
                    [
                        self.data[k]["translation"].strip()
                        for k in indexes_to_merge
                    ]
                ),
            }
            new_data.append(new_dict)
            i = j
            j += 1
        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        audio_path = data["audio_paths"]
        audio_start = data["audio_start"]
        audio_end = data["audio_end"]
        duration = data["duration"]
        transcription = data["transcription"]
        translation = data["translation"]
        y, sr = librosa.load(
            audio_path, offset=audio_start, duration=duration, sr=None
        )
        audio = torch.tensor(y)
        task = "ST"

        source_language = self.source
        target_language = self.target

        return {
            "audio": audio,
            "gt": translation,
            "task": task,
            "transcription": transcription,
            "source_language": source_language,
            "target_language": target_language,
            "sr": sr,
        }

    def get_dataloader(self, batch_size=1, shuffle=True):
        def collate_fn(instances):
            batch = {}
            keys = [
                "audio",
                "gt",
                "task",
                "transcription",
                "source_language",
                "target_language",
                "sr",
            ]
            for key in keys:
                batch[key] = []

            for instance in instances:
                for key in instance:
                    if key not in keys:
                        continue
                    batch[key].append(instance[key])

            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )


if __name__ == "__main__":
    dataset = EuroParlST(
        os.getenv("DATA_HOME"),
        "en_de",
        split="test",
        aim_for_min_duration=60,
        filter_for_data_beetween=(50, 65),
    ).get_dataloader(batch_size=3, shuffle=False)
    for i, batch in enumerate(dataset):
        print(batch)
        if i == 0:
            break

    if False:
        dataset = JamescalamYoutubeEvaluationDataset(
            os.getenv("DATA_HOME")
        ).get_dataloader(batch_size=3, shuffle=False)
        for i, batch in enumerate(dataset):
            #  print(batch)
            if i == 0:
                break

        config_path = os.path.join(
            os.getenv("SPEECHLMM_ROOT"),
            "speechlmm/eval/data_configs/covost_de_en_100_percent.yml",
        )
        data_args = DataArguments(
            data_config_path=config_path,
            is_multimodal=True,
            dataloader_debug=False,
        )
        all_datasets = DatasetsWrapper(data_args)
        dataloader = get_dataloader(
            all_datasets.test_dataset, batch_size=10, shuffle=False
        )

        for i, batch in enumerate(dataloader):
            print(batch)
            if i == 0:
                break

        dataset = MustCEvaluationServerDataset(
            os.getenv("DATA_HOME"), "en_de"
        ).get_dataloader(batch_size=1, shuffle=False)
        for i, batch in enumerate(dataset):
            print(batch)
            if i == 0:
                break
