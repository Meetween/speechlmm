import argparse
import os
import random
import tempfile
from argparse import RawTextHelpFormatter

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import transformers
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from tqdm import tqdm

from speechlmm.eval.utils.tts.export import export_metrics
from speechlmm.eval.utils.tts.generic_utils import (
    compute_cer,
    normalize_text,
    torch_rms_norm,
)
from speechlmm.eval.utils.tts.stt import WhisperSTT


def load_audio(path):
    audio, sr = torchaudio.load(path)
    return audio.mean(dim=0, keepdim=True), sr


# set seed to ensures reproducibility
def set_seed(random_seed=1234):
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    transformers.set_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)


set_seed()

CUDA_AVAILABLE = torch.cuda.is_available()
device = "cuda" if CUDA_AVAILABLE else "cpu"

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

## SECS utils
# automatically checks for cached file, optionally set `cache_dir` location
ecapa2_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
ecapa2 = torch.jit.load(ecapa2_file, map_location="cpu").to(device)
# transcriber = FasterWhisperSTT("large-v3", use_cuda=True)
transcriber = WhisperSTT("large-v3")  # use cuda True if you have a GPU


def get_ecapa2_spk_embedding(path, ref_dBFS=None, model_sr=16000):
    audio, sr = load_audio(path)
    # sample rate of 16 kHz expected
    if sr != model_sr:
        audio = torchaudio.functional.resample(audio, sr, model_sr)

    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    if ref_dBFS is not None:
        audio = torch_rms_norm(audio, db_level=ref_dBFS)

    # compute speaker embedding
    embed = ecapa2(audio.to(device))
    # ensures that l2 norm is applied on output
    embed = torch.nn.functional.normalize(embed, p=2, dim=1)
    return embed.cpu().detach().squeeze().numpy()


## UTMOS utils

# uses UTMOS (https://arxiv.org/abs/2204.02152) Open source (https://github.com/tarepan/SpeechMOS) following https://arxiv.org/abs/2311.12454
mos_predictor = torch.hub.load(
    "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
).to(device)


def compute_UTMOS(audio, ref_dBFS, sr):
    # audio, sr = torchaudio.load(path)
    # audio = torch.from_numpy(audio).unsqueeze(0)
    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    if ref_dBFS is not None:
        audio = torch_rms_norm(audio, db_level=ref_dBFS)
    # predict UTMOS
    score = mos_predictor(audio.to(device), sr).item()
    return score


def compute_metrics(
    tts_wav_path,
    ref_wav_path,
    gt_text,
    ref_speaker_embedding=None,
    language="en",
    debug=False,
    ref_dBFS=None,
):
    tts_wav, sr = load_audio(tts_wav_path)

    language = language.split("-")[0]  # remove the region
    transcription = transcriber.transcribe_audio(
        tts_wav.squeeze(0).numpy(), language=language
    )

    # normalize texts - removing ponctuations
    gt_text_normalized = normalize_text(gt_text)
    transcription_normalized = normalize_text(transcription)

    # compute WER
    cer_tts = compute_cer(gt_text_normalized, transcription_normalized) * 100

    # compute UTMOS
    mos = compute_UTMOS(tts_wav, ref_dBFS, sr)

    # compute SECS using ECAPA2 model
    gen_speaker_embedding = get_ecapa2_spk_embedding(tts_wav_path, ref_dBFS)
    if ref_speaker_embedding is None:
        ref_speaker_embedding = get_ecapa2_spk_embedding(
            ref_wav_path, ref_dBFS
        )
    gt_speaker_embedding = torch.FloatTensor(ref_speaker_embedding).unsqueeze(
        0
    )
    gen_speaker_embedding = torch.FloatTensor(gen_speaker_embedding).unsqueeze(
        0
    )
    secs = torch.nn.functional.cosine_similarity(
        gt_speaker_embedding, gen_speaker_embedding
    ).item()

    if debug:
        print("Speaker Reference Path", ref_wav_path)
        print("TTS Audio Path:", tts_wav_path)
        print("Language:", language)
        print("GT text:", gt_text_normalized)
        print("Transcription:", transcription_normalized)
        print("CER:", cer_tts)
        print("UTMOS MOS:", mos)
        print("SECS:", secs)

    meta_dict = {
        "CER": cer_tts,
        "UTMOS": mos,
        "SECS": secs,
        "Language": language,
    }
    meta_dict_full = meta_dict.copy()
    meta_dict_full["Speaker Reference Path"] = str(ref_wav_path)
    meta_dict_full["Audio Path"] = str(tts_wav_path)
    meta_dict_full["Num. Chars"] = len(gt_text)
    meta_dict_full["Ground Truth Text"] = gt_text
    meta_dict_full["Transcription"] = transcription
    meta_dict_full["Ground Truth Normalized Text"] = gt_text_normalized
    meta_dict_full["Normalized Transcription"] = transcription_normalized
    return meta_dict, meta_dict_full
