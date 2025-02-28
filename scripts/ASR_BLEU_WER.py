# this script tests the quality of ASR through BLEU and WER scores.
# input both for references and prediction is supposed to be a JSONL file, with
# each line containing {"transcript": "text"} as the only field


import argparse
import json
import os
import sys

import jiwer

# from nemo_text_processing.text_normalization.normalize import Normalizer
import nltk
from nltk.translate.bleu_score import corpus_bleu

transformation = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="ASR BLEU and WER")
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to the reference JSONL file",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        required=True,
        help="Path to the hypothesis JSONL file",
    )
    return parser.parse_args()


def calculate_bleu(references, hypotheses):
    # Tokenize the sentences using nltk
    tokenized_refs = [
        [nltk.word_tokenize(ref)] for ref in references
    ]  # BLEU expects a list of references
    tokenized_hyps = [nltk.word_tokenize(hyp) for hyp in hypotheses]

    # Calculate BLEU score
    return corpus_bleu(tokenized_refs, tokenized_hyps)


def main():
    #   normalizer = Normalizer(input_case='cased', lang='en')
    args = parse_args()
    # load both files into memory
    with open(args.ref, "r") as ref_file:
        ref_lines = [json.loads(line) for line in ref_file]
    with open(args.hyp, "r") as hyp_file:
        hyp_lines = [json.loads(line) for line in hyp_file]
    assert len(ref_lines) == len(hyp_lines)
    # check if the ids are the same
    tot_wer = 0
    ref_texts = []
    hyp_texts = []
    # order the hypothesis and references by id
    hyp_lines = sorted(hyp_lines, key=lambda x: x["id"])
    ref_lines = sorted(ref_lines, key=lambda x: x["id"])
    for ref, hyp in zip(ref_lines, hyp_lines):
        assert ref["id"] == hyp["id"]
        ref = transformation(ref["transcript"])
        hyp = transformation(hyp["transcript"])
        # standardize the transcript
        # remove punctuation for this dataset
        # ref = normalizer.normalize(ref, verbose=True, punct_post_process=True)
        # hyp = normalizer.normalize(hyp, verbose=True, punct_post_process=True)
        ref_texts.append(ref)
        hyp_texts.append(hyp)
        tot_wer += jiwer.wer(ref, hyp)
    #        print(ref,hyp)
    #        print("-------------------")
    # compute WER
    word_error_rate = tot_wer / len(hyp_lines)
    # print wer as %
    print(f"WER: {word_error_rate:.2%}")
    bleu_score = calculate_bleu(ref_texts, hyp_texts)
    # print BLEU score multiplied by 100 as done often
    print(f"BLEU: {bleu_score*100:.2f}")


# Ours
# WER: 7.58%
# BLEU: 89.76

# Whisper-v3
# WER: 8.02%
# BLEU: 89.02


if __name__ == "__main__":
    main()
