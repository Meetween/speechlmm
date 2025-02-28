import argparse
import contextlib
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file


def average_checkpoints(args):
    if args.output_folder is None:
        args.output_folder = args.checkpoints_folder.with_name(
            f"{args.checkpoints_folder.name}-avg-last-{args.num_avg}"
        )
    args.output_folder.mkdir(parents=True, exist_ok=True)

    all_checkpoint_dirs = sorted(
        args.checkpoints_folder.glob("checkpoint-*"),
        key=lambda x: int(x.stem.split("-")[1]),
        reverse=True,
    )
    for checkpoint_file in [
        "mm_audio_adapter.bin",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
    ]:
        averaged_checkpoint = {}
        for dir_ in all_checkpoint_dirs[: args.num_avg]:
            checkpoint_path = dir_ / checkpoint_file
            if not checkpoint_path.exists():
                continue

            load_fn = (
                load_file
                if checkpoint_file.endswith(".safetensors")
                else torch.load
            )
            checkpoint = load_fn(checkpoint_path)
            for key in checkpoint.keys():
                if key not in averaged_checkpoint:
                    averaged_checkpoint[key] = checkpoint[key] / args.num_avg
                else:
                    averaged_checkpoint[key] += checkpoint[key] / args.num_avg

            last_config = dir_ / "config.json"
            last_adapter_config = dir_ / "adapter_config.json"
            del checkpoint

        if len(averaged_checkpoint) > 0:
            torch.save(
                averaged_checkpoint, args.output_folder / checkpoint_file
            )

    # copy also the config files
    with contextlib.suppress(FileNotFoundError):
        shutil.copy(last_config, args.output_folder)
    with contextlib.suppress(FileNotFoundError):
        shutil.copy(last_adapter_config, args.output_folder)

    print(f"Saved averaged checkpoint to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-folder", type=Path, required=True)
    parser.add_argument("--num-avg", type=int, default=5)
    parser.add_argument("--output-folder", type=Path, default=None)
    average_checkpoints(parser.parse_args())
