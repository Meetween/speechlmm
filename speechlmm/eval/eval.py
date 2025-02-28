import argparse
import json
import os

from speechlmm.arguments import DataArguments
from speechlmm.dataset.datasets_wrapper import DatasetsWrapper
from speechlmm.eval.evaluator import Evaluator
from speechlmm.eval.metrics import compute_asr_metrics, compute_st_metrics

# import from utils in the same folder as this (speechlmm/eval/utils)
from speechlmm.eval.utils.eval_dataloader import (
    JamescalamYoutubeEvaluationDataset,
    MustCEvaluationServerDataset,
    OldFormatLibriTTSLibriSpeechDataset,
    get_dataloader,
)
from speechlmm.mm_utils import get_model_name_from_path
from speechlmm.model.builder import load_pretrained_model
from speechlmm.utils import disable_torch_init


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--modality", type=str, default="audio")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--from-yml", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--tokenizer-padding-side", type=str, default="left")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    assert args.model_path is not None
    assert args.results_dir is not None
    assert args.dataset is not None

    # also put into lower case the task and dataset
    args.dataset = args.dataset.lower()

    return args


def get_preds_path(results_dir, dataset):
    # If there is already a file with the same name, add a number to the end of the file name to avoid overwriting
    dataset_name = dataset.split("/")[-1].split(".")[0]
    results_path = os.path.join(results_dir, f"{dataset_name}_preds.json")
    if os.path.exists(results_path):
        i = 1
        while os.path.exists(results_path):
            results_path = os.path.join(
                results_dir, f"{dataset_name}_{i}_preds.json"
            )
            i += 1
    return results_path


def get_scores_path(results_dir, dataset):
    # If there is already a file with the same name, add a number to the end of the file name to avoid overwriting
    dataset_name = dataset.split("/")[-1].split(".")[0]
    results_path = os.path.join(results_dir, f"{dataset_name}_scores.json")

    if os.path.exists(results_path):
        i = 1
        while os.path.exists(results_path):
            results_path = os.path.join(
                results_dir, f"{dataset_name}_{i}_scores.json"
            )
            i += 1
    return results_path


def write_preds(results_dict, results_dir, dataset):
    json_results_path = get_preds_path(results_dir, dataset)
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

    with open(json_results_path, "w", encoding="utf-8") as json_file:
        json.dump(results_list, json_file, ensure_ascii=False, indent=4)


def write_scores(all_metrics, results_dir, dataset):
    json_scores_path = get_scores_path(results_dir, dataset)
    with open(json_scores_path, "w", encoding="utf-8") as json_file:
        json.dump(all_metrics, json_file, ensure_ascii=False, indent=4)


def main():
    args = args_parser()
    with open(args.model_path + "/config.json") as f:
        config = json.load(f)
        model_base = config["_name_or_path"]

    results_dir = os.path.join(args.results_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.from_yml:
        data_config_path = args.dataset
        data_args = DataArguments(
            data_config_path=data_config_path,
            is_multimodal=True,
            dataloader_debug=False,
        )
        dataset = DatasetsWrapper(data_args)
        dataloader = get_dataloader(
            dataset.test_dataset, batch_size=args.batch_size, shuffle=False
        )
    elif args.dataset in [
        "librispeech_test_clean",
        "librispeech_test_other",
        "libritts_300_test_clean",
    ]:
        data_dir = os.path.join(args.data_dir)
        dataloader = OldFormatLibriTTSLibriSpeechDataset(
            data_dir, args.dataset
        ).get_dataloader(batch_size=args.batch_size, shuffle=False)
    elif "mustc_evaluation_server" in args.dataset:
        # mustc_evaluation_server_{direction} this is how it should be. be careful since direction can be a single language (en, de) or a pair (en_de)
        direction = args.dataset.split("mustc_evaluation_server_")[-1]
        data_dir = os.path.join(args.data_dir)
        dataloader = MustCEvaluationServerDataset(
            data_dir, direction
        ).get_dataloader(batch_size=args.batch_size, shuffle=False)
    elif "youtube_asr_evaluation_chunk_size" in args.dataset:
        chunk_size = int(
            args.dataset.split("youtube_asr_evaluation_chunk_size_")[-1]
        )
        data_dir = os.path.join(args.data_dir)
        dataloader = JamescalamYoutubeEvaluationDataset(
            data_dir, chunk_size
        ).get_dataloader(batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    assert (
        args.modality is not None
        and args.modality == "image"
        or args.modality == "audio"
    ), "modality must be either 'image' or 'audio'"

    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path,
        model_base,
        model_name,
        modality=args.modality,
        device=args.device,
    )

    llama_3_1 = False

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "vicuna_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "llama-3" in model_name.lower():
        if not "llama-3.1" in model_name.lower():
            raise ValueError(
                "Llama-3 is deprecated, please use Llama-3.1 instead."
            )
        conv_mode = "llama_3_1"
        llama_3_1 = True
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    tokenizer.padding_side = args.tokenizer_padding_side
    tokenizer.pad_token = tokenizer.eos_token  # pad token is eos token

    if tokenizer.model_max_length >= 100_000:
        tokenizer.model_max_length = 4096  # FIXME: hardcoded atm, due to Mistral HF tokenizer model_max_length = 1000000000000000019884624838656
    model.config.tokenizer_padding_side = args.tokenizer_padding_side
    if "llama-3" in model_name.lower():
        model.generation_config.eos_token_id = 128009
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        processor=processor,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        conv_mode=args.conv_mode,
    )

    results_dict = evaluator.generate()
    write_preds(results_dict, results_dir, args.dataset)

    all_metrics = Evaluator.compute_metrics(results_dict)
    write_scores(all_metrics, results_dir, args.dataset)


if __name__ == "__main__":
    main()
