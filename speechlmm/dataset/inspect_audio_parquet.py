import argparse
import os
import random
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


def sanitize_path_to_name(path):
    """Convert a full file path to a safe name by replacing unsafe characters."""
    safe_name = (
        str(path).replace("/", "_").replace("\\", "_").replace(":", "_")
    )
    safe_name = safe_name.lstrip("_")
    safe_name = safe_name.replace(".", "_")
    return safe_name


def get_unique_directory(base_path):
    """Create a unique directory path by appending a number if necessary"""
    if not os.path.exists(base_path):
        return base_path

    counter = 1
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def analyze_value(value):
    """Helper function to analyze a value's type and shape/length"""
    MAX_LENGTH = 500  # Maximum length of string representation
    if isinstance(value, dict):
        return {
            k: (
                (
                    str(v)[:MAX_LENGTH] + "..."
                    if len(str(v)) > MAX_LENGTH
                    else str(v)
                ),
                f"{type(v)} - Shape/Length: {np.shape(v) if hasattr(v, 'shape') else len(v) if hasattr(v, '__len__') else 'N/A'}",
            )
            for k, v in value.items()
        }
    else:
        return (
            (
                str(value)[:MAX_LENGTH] + "..."
                if len(str(value)) > MAX_LENGTH
                else str(value)
            ),
            f"{type(value)} - Shape/Length: {np.shape(value) if hasattr(value, 'shape') else len(value) if hasattr(value, '__len__') else 'N/A'}",
        )


def analyze_dataset(dataset, output_dir, random_seed, num_samples=5):
    """Analyze dataset and save results"""
    random.seed(random_seed)
    info_path = os.path.join(output_dir, "dataset_info.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(info_path, "w") as f:
        for ds_key in dataset.keys():
            f.write(f"\nDataset: {ds_key}\n")
            dataset_size = len(dataset[ds_key])
            sample_indices = random.sample(
                range(dataset_size), min(num_samples, dataset_size)
            )
            f.write(f"\nDataset size: {dataset_size} samples\n")
            f.write("\nFields available:\n")
            for field_name, field_value in dataset[ds_key].features.items():
                f.write(f"- {field_name} (Type: {type(field_value)})\n")
            f.write(f"\nAnalyzing {len(sample_indices)} random samples:\n")
            for i, idx in enumerate(sample_indices):
                f.write(f"\n{'='*50}\n")
                f.write(f"Sample Index: {idx}\n")
                f.write(
                    f"Audio names: {ds_key}_sample_{i+1}_[NAME_OF_AUDIO_FIELD].wav\n"
                )
                for field_name, field_value in dataset[
                    ds_key
                ].features.items():
                    f.write(f"\nField: {field_name}\n")
                    try:
                        example = dataset[ds_key][idx][field_name]
                        analysis = analyze_value(example)
                        if isinstance(analysis, dict):
                            f.write("Dictionary contents:\n")
                            for k, (value_str, type_info) in analysis.items():
                                f.write(f"  {k}:\n")
                                f.write(f"    Value: {value_str}\n")
                                f.write(f"    Info: {type_info}\n")
                        else:
                            value_str, type_info = analysis
                            f.write(f"Value: {value_str}\n")
                            f.write(f"Info: {type_info}\n")
                    except Exception as e:
                        error_msg = f"Error accessing value: {str(e)}"
                        print(error_msg)
                        f.write(f"{error_msg}\n")


def sample_and_save_audio(dataset, output_dir, num_samples, random_seed):
    """Sample random audio files from a dataset and save them to disk.

    Identifies audio fields by checking for dictionary fields containing 'array' and
    'sampling_rate' keys.
    """
    random.seed(random_seed)
    audio_dir = os.path.join(output_dir, "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)

    # Identify datasets and corresponding audio keys
    datasets_info = {}
    for ds_key in dataset.keys():
        if len(dataset[ds_key]) == 0:
            continue
        first_item = dataset[ds_key][0]
        audio_keys = []
        for key, value in first_item.items():
            if isinstance(value, dict):
                if "array" in value and "sampling_rate" in value:
                    audio_keys.append(key)
        if audio_keys:
            datasets_info[ds_key] = audio_keys

    if not datasets_info:
        print("\nNo audio fields found in dataset")
        return

    for ds_key in datasets_info:
        dataset_size = len(dataset[ds_key])
        random_indices = random.sample(
            range(dataset_size), min(num_samples, dataset_size)
        )
        for i, idx in enumerate(random_indices):
            for audio_key in datasets_info[ds_key]:
                feature_value = dataset[ds_key][idx][audio_key]
                audio_data = feature_value["array"]
                sampling_rate = feature_value["sampling_rate"]
                output_path = os.path.join(
                    audio_dir, f"{ds_key}_sample_{i+1}_{audio_key}.wav"
                )
                sf.write(output_path, audio_data, sampling_rate)


def compute_audio_stats(dataset, output_dir, random_seed, max_samples=300):
    """
    Sample audio files from the dataset (up to max_samples samples per audio field or all available if fewer)
    and compute statistics including duration quantiles, waveform mean, waveform std,
    waveform absolute mean, and mel spectrogram features (mean and std) for each audio field.
    Saves the results in 'audio_stats.txt' within the output directory.
    """
    print("\nComputing audio statistics (--audio-stats parameter)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fft = 1024
    win_length = 1024
    hop_length = 256
    n_mels = 100

    # Identify audio fields in the dataset
    audio_fields = {}
    for ds_key in dataset.keys():
        if len(dataset[ds_key]) == 0:
            continue
        first_item = dataset[ds_key][0]
        keys = []
        for key, value in first_item.items():
            if (
                isinstance(value, dict)
                and "array" in value
                and "sampling_rate" in value
            ):
                keys.append(key)
        if keys:
            audio_fields[ds_key] = keys

    if not audio_fields:
        print("No audio fields found for stats computation")
        return

    stats_str = "Audio Statistics\n\n"

    # Process each dataset and audio field separately
    for ds_key, keys in audio_fields.items():
        for audio_key in keys:
            print(f"\nProcessing audio field: {ds_key}/{audio_key}")

            # Accumulators for statistics for this field
            durations = []
            waveform_means = []
            waveform_stds = []
            waveform_abs_means = []
            mel_means = []
            mel_stds = []

            dataset_size = len(dataset[ds_key])
            sample_count = (
                dataset_size
                if max_samples == "all"
                else min(dataset_size, max_samples)
            )

            # Generate random indices for this field
            rng = random.Random(random_seed)
            indices = list(range(dataset_size))
            rng.shuffle(indices)
            indices = indices[:sample_count]

            pbar = tqdm(
                total=sample_count, desc=f"Processing {ds_key}/{audio_key}"
            )

            for idx in indices:
                feature_value = dataset[ds_key][idx][audio_key]
                audio_data = feature_value["array"]
                sample_rate_val = feature_value["sampling_rate"]

                # Convert the numpy array into a tensor
                waveform = torch.tensor(audio_data, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)  # shape: (1, T)
                orig_length = waveform.shape[1]

                # Compute waveform statistics
                durations.append(orig_length / sample_rate_val)
                waveform_means.append(waveform.mean().item())
                waveform_stds.append(waveform.std().item())
                waveform_abs_means.append(waveform.abs().mean().item())

                # Compute mel spectrogram
                waveform = waveform.to(device)
                mel_transform = MelSpectrogram(
                    sample_rate=sample_rate_val,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=n_mels,
                ).to(device)
                mel_spec = mel_transform(waveform)
                mel_means.append(mel_spec.mean().item())
                mel_stds.append(mel_spec.std().item())

                pbar.update(1)

            pbar.close()

            # Compute statistics for this field
            durations_np = np.array(durations)
            quantiles = np.percentile(
                durations_np, [0, 25, 50, 75, 100]
            ).tolist()

            # Calculate total duration
            total_duration_seconds = np.sum(durations_np)
            hours = int(total_duration_seconds // 3600)
            minutes = int((total_duration_seconds % 3600) // 60)
            seconds = total_duration_seconds % 60

            stats_str += f"Field: {ds_key}/{audio_key}\n"
            stats_str += f"Number of samples analyzed: {len(durations)} out of {dataset_size} total files\n"
            stats_str += f"Total duration: {hours}h {minutes}m {seconds:.2f}s ({total_duration_seconds:.2f} seconds)\n"
            stats_str += (
                f"Duration quantiles (0%, 25%, 50%, 75%, 100%): {quantiles}\n"
            )
            stats_str += f"Waveform mean: {np.mean(waveform_means):.8f}\n"
            stats_str += f"Waveform std: {np.mean(waveform_stds):.8f}\n"
            stats_str += (
                f"Waveform abs mean: {np.mean(waveform_abs_means):.8f}\n"
            )
            stats_str += f"Mel spectrogram mean: {np.mean(mel_means):.8f}\n"
            stats_str += f"Mel spectrogram std: {np.mean(mel_stds):.8f}\n\n"

    stats_path = os.path.join(output_dir, "audio_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)
    print("Audio statistics computed and saved at:", stats_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a Parquet dataset and sample audio if present"
    )
    parser.add_argument("path", type=str, help="Path to the Parquet file")
    parser.add_argument(
        "num_samples",
        type=int,
        nargs="?",
        default=5,
        help="Number of audio samples to save (default: 5)",
    )
    parser.add_argument(
        "--audio-stats",
        action="store_true",
        help="Compute audio statistics for a subset of audios",
    )
    parser.add_argument(
        "--max-stats-samples",
        type=str,
        default=300,
        help="Maximum number of samples to use for audio statistics (default: 300, use 'all' to process entire dataset)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: File not found at {args.path}")
        return

    try:
        print(f"Loading dataset from {args.path}")
        dataset = datasets.load_dataset("parquet", data_files=args.path)
        dataset_name = sanitize_path_to_name(args.path)
        base_results_dir = os.path.join("samples_from_dataset", dataset_name)
        results_dir = get_unique_directory(base_results_dir)
        os.makedirs(results_dir)
        print(f"Saving results to: {results_dir}")

        random_seed = random.randint(0, 2**42)
        analyze_dataset(dataset, results_dir, random_seed, args.num_samples)
        sample_and_save_audio(
            dataset, results_dir, args.num_samples, random_seed
        )
        if args.audio_stats:
            if args.max_stats_samples != "all":
                try:
                    args.max_stats_samples = int(args.max_stats_samples)
                except ValueError:
                    print(
                        f"Invalid max-stats-samples value: {args.max_stats_samples}. Using default of 300."
                    )
                    args.max_stats_samples = 300
            compute_audio_stats(
                dataset, results_dir, random_seed, args.max_stats_samples
            )
        full_path_results_dir = Path(results_dir).resolve()
        print(
            f"\nAnalysis complete! Results saved in: {full_path_results_dir}"
        )
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")


if __name__ == "__main__":
    main()
