import argparse

from speechlmm.mm_utils import get_model_name_from_path
from speechlmm.model.builder import load_pretrained_model


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map="cpu",
        modality=args.modality,
    )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)


# python scripts/merge_lora_weights.py     --model-path $SPEECHLMM_ROOT/checkpoints/lora-speechlmm-v1.5-7b_finetune/
#     --model-base lmsys/vicuna-7b-v1.5     --save-model-path $SPEECHLMM_ROOT/merge-lora-finetune --modality audio

# python scripts/merge_lora_weights.py     --model-path $SPEECHLMM_ROOT/checkpoints/lora-speechlmm-v1.5-7b_finetune_asr_sqa/  \
# --model-base lmsys/vicuna-7b-v1.5     --save-model-path $SPEECHLMM_ROOT/merge-lora-finetune-asr-sqa --modality audio
