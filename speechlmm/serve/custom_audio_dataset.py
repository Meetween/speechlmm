from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

from speechlmm.mm_utils import monify_and_resample_audio
from speechlmm.serve.prompt_utils import PromptBuilder


def load_audio_into_tensor(
    audio_path: str, target_sr: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    if "'" in audio_path:
        audio_path = audio_path.replace("'", "")
    audio, orig_sr = torchaudio.load(
        audio_path, normalize=True, channels_first=True
    )
    return monify_and_resample_audio(audio, orig_sr, target_sr)


# NOTE: We implement this cause we need left padding,
# but since we are using torch==2.3.1 the argument `padding_side`
# is not implemented in the torch.nn.utils.rnn.pad_sequence
def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int,
    padding_side: str = "left",
):
    """Pad a list of variable length Tensors to a single Tensor with padding on the left.

    Args:
        sequences (List[torch.Tensor]): List of sequences to pad
        batch_first (bool): Output batch dimension first if True
        padding_value (int): Value for padded elements
        padding_side (str): Side to add padding ('left' or 'right')

    Returns:
        torch.Tensor: Padded tensor
    """
    sequences = [seq.squeeze(0) for seq in sequences]
    max_len = max([len(seq) for seq in sequences])
    batch_size = len(sequences)

    out_dims = (batch_size, max_len)
    out_tensor = torch.full(
        out_dims,
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )

    for batch_idx, seq in enumerate(sequences):
        if padding_side == "left":
            out_tensor[batch_idx, -len(seq) :] = seq
        else:
            out_tensor[batch_idx, : len(seq)] = seq

    return out_tensor


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_path: str,
        prompt_builder: PromptBuilder,
        target_sr: Optional[int] = None,
    ):
        self.prompt_builder = prompt_builder
        self.target_sr = target_sr
        self.audio_files = []

        # Handle both single file and directory
        path = Path(audio_path)
        if path.is_file():
            self.audio_files = [str(path)]
        elif path.is_dir():
            # Get all audio files with common extensions
            audio_extensions = [".wav", ".mp3"]
            for ext in audio_extensions:
                self.audio_files.extend([str(p) for p in path.glob(f"*{ext}")])
        else:
            raise ValueError(
                f"Path {audio_path} is neither a file nor a directory"
            )

        self.audio_files.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = load_audio_into_tensor(audio_path, self.target_sr)
        prompt = self.prompt_builder.get_prompt()
        input_ids = self.prompt_builder.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids
        return {
            "audio": audio,
            "input_ids": input_ids,
            "sampling_rate": sr,
            "audio_path": audio_path,
            "prompt": prompt,
            "task": self.prompt_builder.task,
        }

    def get_data_loader(self, batch_size=1, shuffle=False):
        def collate_fn(instances):
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = pad_sequence(
                input_ids,
                padding_value=self.prompt_builder.tokenizer.pad_token_id,
                padding_side="left",
            )
            prompts = [instance["prompt"] for instance in instances]
            batch = dict(
                input_ids=input_ids,
                prompts=prompts,
                attention_mask=input_ids.ne(
                    self.prompt_builder.tokenizer.pad_token_id
                ),
            )
            audios = [instance["audio"] for instance in instances]
            srs = [instance["sampling_rate"] for instance in instances]
            batch["audios_srs"] = list(zip(audios, srs))
            batch["prompt"] = [instance["prompt"] for instance in instances]
            batch["task"] = [instance["task"] for instance in instances]
            batch["audio_path"] = [
                instance["audio_path"] for instance in instances
            ]
            return batch

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
