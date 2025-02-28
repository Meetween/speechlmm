import argparse
import json
import ssl
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speechlmm.eval.utils.tts.compute_metrics import compute_metrics

ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-file", type=Path, required=True)
    args = parser.parse_args()

    scores = {}
    global_scores = {
        "CER": [],
        "UTMOS": [],
        "SECS": [],
    }
    for audio_file in tqdm(list(args.audio_dir.glob("*.wav"))):
        if "CONDITION" in audio_file.stem:
            continue
        ref_wav = str(audio_file).replace(
            audio_file.name, f"VOICE_CONDITIONING.wav"
        )
        ref_text = str(audio_file).replace(
            audio_file.name, f"{audio_file.stem}.txt"
        )
        gt_text = Path(ref_text).read_text().strip()
        # tts_wav, sr = torchaudio.load(audio_file, normalize=True)
        meta_dict, meta_dict_full = compute_metrics(
            audio_file, ref_wav, gt_text, language="en", debug=True
        )
        scores[audio_file.stem] = meta_dict_full
        global_scores["CER"].append(meta_dict["CER"])
        global_scores["UTMOS"].append(meta_dict["UTMOS"])
        global_scores["SECS"].append(meta_dict["SECS"])

    # print global scores
    # CER
    mean_cer = np.mean(global_scores["CER"])
    std_cer = np.std(global_scores["CER"])
    print(f"Mean CER: {mean_cer:.2f} (± {std_cer:.2f})")
    # SECS
    mean_secs = np.mean(global_scores["SECS"])
    std_secs = np.std(global_scores["SECS"])
    print(f"Mean SECS: {mean_secs:.2f} (± {std_secs:.2f})")
    # UTMOS
    mean_utmos = np.mean(global_scores["UTMOS"])
    std_utmos = np.std(global_scores["UTMOS"])
    print(f"Mean UTMOS: {mean_utmos:.2f} (± {std_utmos:.2f})")

    args.output_file.write_text(json.dumps(scores, indent=4))
    print(f"Results saved to {args.output_file}")

# python speechlmm/eval/eval_TTS.py \
# -a $SCRATCH/generated/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io_moshi_bert_punctuaction_checkpoint-35000 \
# -o $SCRATCH/generated/speechlmm-pretrain-audio-llama_3_1-moshi_bert-qformer-features-audio_io_moshi_bert_punctuaction_checkpoint-35000/metrics.json
