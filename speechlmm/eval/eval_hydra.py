import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from speechlmm.arguments import DataArguments
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.eval.evaluator import Evaluator
from speechlmm.eval.utils.eval_dataloader import (
    JamescalamYoutubeEvaluationDataset,
    MustCEvaluationServerDataset,
    OldFormatLibriTTSLibriSpeechDataset,
    get_dataloader,
)
from speechlmm.model.builder import load_pretrained_model
from speechlmm.utils import disable_torch_init


def get_custom_dataset(dataset_name: str, datasets_dir: str):
    if dataset_name in [
        "librispeech_test_clean",
        "librispeech_test_other",
        "libritts_300_test_clean",
    ]:
        dataset = OldFormatLibriTTSLibriSpeechDataset(
            datasets_dir, dataset_name
        )
    elif "mustc_evaluation_server" in dataset_name:
        direction = dataset_name.split("mustc_evaluation_server_")[-1]
        dataset = MustCEvaluationServerDataset(datasets_dir, direction)
    elif "youtube_asr_evaluation_chunk_size" in dataset_name:
        chunk_size = int(
            dataset_name.split("youtube_asr_evaluation_chunk_size_")[-1]
        )
        dataset = JamescalamYoutubeEvaluationDataset(datasets_dir, chunk_size)
    else:
        raise ValueError(f"Unknown custom dataset: {dataset_name}")

    return dataset


def get_results_path(dataset_name_or_path, results_dir, results_name):
    # If there is already a file with the same name, add a number to the
    # end of the file name to avoid overwriting
    results_dir = Path(results_dir)
    dataset_name = Path(dataset_name_or_path).stem
    results_path = results_dir / f"{dataset_name}_{results_name}.json"

    i = 1
    while results_path.exists():
        results_path = results_dir / f"{dataset_name}_{i}_{results_name}.json"
        i += 1

    return results_path


def get_preds_path(dataset_name_or_path, results_dir):
    return get_results_path(dataset_name_or_path, results_dir, "preds")


def get_scores_path(dataset_name_or_path, results_dir):
    return get_results_path(dataset_name_or_path, results_dir, "scores")


def write_preds(results_dict, results_dir, dataset):
    json_preds_path = get_preds_path(
        dataset_name_or_path=dataset, results_dir=results_dir
    )
    results_list = []
    for (
        task,
        source_lang,
        target_lang,
        prompt,
        gt,
        pred,
        transcription,
    ) in zip(
        results_dict["tasks"],
        results_dict["source_langs"],
        results_dict["target_langs"],
        results_dict["prompts"],
        results_dict["gts"],
        results_dict["preds"],
        results_dict["transcriptions"],
    ):
        results_list.append(
            {
                "prompt": prompt,
                "transcription": transcription,
                "predictions": pred,
                "groundtruth": gt,
                "task": task,
                "source_language": source_lang,
                "target_language": target_lang,
            }
        )

    with open(json_preds_path, "w", encoding="utf-8") as json_file:
        json.dump(results_list, json_file, ensure_ascii=False, indent=4)


def write_scores(all_metrics, results_dir, dataset):
    json_scores_path = get_scores_path(
        dataset_name_or_path=dataset, results_dir=results_dir
    )
    with open(json_scores_path, "w", encoding="utf-8") as json_file:
        json.dump(all_metrics, json_file, ensure_ascii=False, indent=4)


@hydra.main(
    version_base=None,
    config_path="../../conf/speechlmm",
    config_name="evaluate",
)
def eval(config: DictConfig) -> None:
    if not config:
        raise ValueError(
            "Config is empty or non-existent. Please make sure to specify a "
            "valid config file via --config-name."
        )

    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if Path(config.eval.dataset_name_or_path).is_file():
        data_args = DataArguments(
            data_config_path=config.eval.dataset_name_or_path,
            is_multimodal=True,
            dataloader_debug=False,
        )
        dataset = DatasetsWrapper(data_args)
        dataloader = get_dataloader(
            dataset.test_dataset,
            batch_size=config.eval.batch_size,
            shuffle=False,
        )
    else:  # `dataset_name_or_path` is the name of a custom dataset
        dataset = get_custom_dataset(
            config.eval.dataset_name_or_path,
            config.eval.datasets_dir,
        )
        dataloader = dataset.get_dataloader(
            batch_size=config.eval.batch_size, shuffle=False
        )

    disable_torch_init()
    torch.set_default_device("cuda")

    model = load_pretrained_model(
        config.model.name_or_path,
        conversation_version=config.model.conversation_version,
        tokenizer_padding_side=config.model.tokenizer_padding_side,
    )
    model.to("cuda")

    # TODO(anferico): make sure I don't need to set
    # eos_token_id = 128009 in the case of Llama 3.1 here (it should be
    # enough to set it in
    # `HfTextDecoder._adapt_tokenizer_to_conversation_version`)
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        tokenizer=model.text_decoder.tokenizer,
        processor=(
            model.audio_encoder.processor
            if getattr(model, "audio_encoder", None) is not None
            else None
        ),
        temperature=config.generation.temperature,
        max_new_tokens=config.generation.max_new_tokens,
        conv_mode=model.text_decoder.conversation_version,
    )

    results_dict = evaluator.generate()
    write_preds(
        results_dict, str(results_dir), config.eval.dataset_name_or_path
    )

    all_metrics = Evaluator.compute_metrics(results_dict)
    write_scores(
        all_metrics, str(results_dir), config.eval.dataset_name_or_path
    )


if __name__ == "__main__":
    eval()
