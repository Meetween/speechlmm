import random
from typing import List, NamedTuple, Optional

import datasets
import evaluate
import jiwer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu


class EvalOnGeneratePrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        sources (`List[str]`): (Optional) Source sentences, used in Speech Translation task for computing Comet scores.
        predictions (`List[str]`): Model outputs decoded to string.
        references (`List[str]`): Ground truth strings.
    """

    predictions: List[str]
    references: List[str]
    sources: Optional[List[str]] = None


class Wer(evaluate.EvaluationModule):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            description="Word Error Rate (WER) is a common metric for evaluating the performance of an automatic speech recognition (ASR) system.",
            citation="""@misc{unknown,
                          author = {},
                          title = {Word Error Rate (WER)},
                          year = {},
                          url = {https://en.wikipedia.org/wiki/Word_error_rate},
                        }""",
            homepage="https://en.wikipedia.org/wiki/Word_error_rate",
            inputs_description="Predictions and references for WER computation.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=["https://en.wikipedia.org/wiki/Word_error_rate"],
        )

    def _compute(self, predictions, references):
        transformation = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.RemoveMultipleSpaces(),
            ]
        )
        transformed_references = [
            transformation(reference) for reference in references
        ]
        transformed_predictions = [
            transformation(prediction) for prediction in predictions
        ]
        # Compute WER scores using transformed texts
        wer_scores = [
            jiwer.wer(ref, pred)
            for ref, pred in zip(
                transformed_references, transformed_predictions
            )
        ]
        wer = sum(wer_scores) / len(wer_scores)
        # wer two decimal places only and to percentage
        wer = round(wer, 4) * 100
        return {"wer": wer}


class Bleu(evaluate.EvaluationModule):
    def __init__(self):
        super().__init__()
        # self.download_and_prepare()

    def _info(self):
        return evaluate.EvaluationModuleInfo(
            description="BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text which has been machine-translated from one natural language to another.",
            citation="""@inproceedings{papineni2002bleu,
                          title={BLEU: a method for automatic evaluation of machine translation},
                          author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
                          booktitle={Proceedings of the 40th annual meeting on association for computational linguistics},
                          pages={311--318},
                          year={2002},
                          organization={Association for Computational Linguistics}
                        }""",
            homepage="https://en.wikipedia.org/wiki/BLEU",
            inputs_description="Predictions and references for BLEU computation.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://aclanthology.org/P02-1040/",
                "https://github.com/mjpost/sacreBLEU",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("punkt")

    def _compute(self, predictions, references):
        transformation = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.RemoveMultipleSpaces(),
            ]
        )
        transformed_references = [
            transformation(reference) for reference in references
        ]
        transformed_predictions = [
            transformation(prediction) for prediction in predictions
        ]
        score = 0
        refs = []
        preds = []
        for ref, pred in zip(transformed_references, transformed_predictions):
            refs.append([ref.split()])
            preds.append(pred.split())
        score = corpus_bleu(refs, preds)
        score = round(score, 4) * 100
        return {"bleu": score}


class SacreBleu(evaluate.EvaluationModule):
    def __init__(self):
        super().__init__()
        # self.download_and_prepare()

    def _info(self):
        return evaluate.EvaluationModuleInfo(
            description="BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text which has been machine-translated from one natural language to another.",
            citation="""@inproceedings{papineni2002bleu,
                          title={BLEU: a method for automatic evaluation of machine translation},
                          author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
                          booktitle={Proceedings of the 40th annual meeting on association for computational linguistics},
                          pages={311--318},
                          year={2002},
                          organization={Association for Computational Linguistics}
                        }""",
            homepage="https://en.wikipedia.org/wiki/BLEU",
            inputs_description="Predictions and references for BLEU computation.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://aclanthology.org/P02-1040/",
                "https://github.com/mjpost/sacreBLEU",
            ],
        )

    def _compute(self, predictions, references):
        score = sacrebleu_corpus_bleu(predictions, [references])
        score = round(score.score, 4)
        return {"bleu": score}


def compute_asr_metrics(eval_preds: EvalOnGeneratePrediction):
    preds, refs = eval_preds.predictions, eval_preds.references
    wer = Wer()
    wer_score = wer.compute(predictions=preds, references=refs)
    return wer_score


def compute_vsr_metrics(eval_preds: EvalOnGeneratePrediction):
    # FIXME check VSR Metrics
    preds, refs = eval_preds.predictions, eval_preds.references
    wer = Wer()
    wer_score = wer.compute(predictions=preds, references=refs)
    return wer_score


def compute_st_metrics(eval_preds: EvalOnGeneratePrediction):
    srcs, preds, refs = (
        eval_preds.sources,
        eval_preds.predictions,
        eval_preds.references,
    )
    bleu = Bleu()
    sacrebleu = SacreBleu()
    bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
    sacrebleu_score = sacrebleu.compute(predictions=preds, references=refs)[
        "bleu"
    ]
    if srcs is None:
        return {"bleu": bleu_score, "sacrebleu": sacrebleu_score}

    comet = evaluate.load("comet")
    comet_score = comet.compute(
        sources=srcs,
        predictions=preds,
        references=refs,
    )["mean_score"]

    comet_score = round(comet_score, 4) * 100

    return {
        "bleu": bleu_score,
        "sacrebleu": sacrebleu_score,
        "comet": comet_score,
    }


# Testing the Wer metric
if __name__ == "__main__":
    metric = SacreBleu()
    predictions = ["this is a cat", "hello world!"]
    references = ["this is a cat", "hello world"]
    print(metric.compute(predictions=predictions, references=references))
