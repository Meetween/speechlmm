from pathlib import Path

from speechlmm.arguments import DataArguments, TrainingArguments
from speechlmm.model.configuration_speechlmm import SpeechLmmConfig


def get_run_tags(
    config: SpeechLmmConfig,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    run_tags = [
        f"text_decoder: {Path(config.text_decoder._name_or_path).name}"
    ]
    if hasattr(config, "vision_encoder"):
        run_tags.extend(
            [
                f"vision_encoder: {Path(config.vision_encoder._name_or_path).name}",
                f"vision_adapter: {config.vision_adapter.model_type}",
            ]
        )
    if hasattr(config, "audio_encoder"):
        run_tags.extend(
            [
                f"audio_encoder: {Path(config.audio_encoder._name_or_path).name}",
                f"audio_adapter: {config.audio_adapter.model_type}",
            ]
        )

    training_mode = (
        "finetune"
        if training_args.pretrained_checkpoint is not None
        else "pretrain"
    )
    run_tags.extend(
        [
            f"data_config: {Path(data_args.data_config_path).stem}",
            f"training_mode: {training_mode}",
        ]
    )

    return run_tags
