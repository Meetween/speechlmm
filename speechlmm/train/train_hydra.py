import logging
import os
import re
import subprocess
import sys
from operator import attrgetter
from pathlib import Path
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


def stem(s: str) -> str:
    return Path(s).stem


OmegaConf.register_new_resolver("stem", stem)

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", logging.INFO))


@hydra.main(
    version_base=None,
    config_path="../../conf/speechlmm",
    config_name="pretrain",
)
def train(config: DictConfig) -> None:
    if not config:
        raise ValueError(
            "Config is empty or non-existent. Please make sure to specify a "
            "valid config file via --config-name."
        )

    config = wire_multimodal_adapters(config)
    config = handle_missing_keys_and_multirun_quirks(config)
    config = handle_adjustments(config)

    # fmt: off
    dataset_loading_command = [
        "python", "speechlmm/dataset/load_dataset.py",
            "--config", str(config.data.data_config_path),
            "--num-proc", str(config.data.num_proc_for_preprocessing),
    ]
    # fmt: on

    if config.data.rebuild_dataset_cache:
        dataset_loading_command.append("--rebuild-cache")
    if config.data.cache_final_datasets:
        dataset_loading_command.append("--cache-final-datasets")
    if config.data.audio_input_sampling_rate is not None:
        dataset_loading_command.extend(
            [
                "--audio-input-sampling-rate",
                str(config.data.audio_input_sampling_rate),
            ]
        )
    if config.data.codec_sampling_rate is not None:
        dataset_loading_command.extend(
            [
                "--codec-sampling-rate",
                str(config.data.codec_sampling_rate),
            ]
        )
    if config.data.codec_frame_rate is not None:
        dataset_loading_command.extend(
            [
                "--codec-frame-rate",
                str(config.data.codec_frame_rate),
            ]
        )

    is_main_process = os.getenv("SLURM_PROCID", "0") == "0"
    if is_main_process:
        logger.info(
            f"Launching dataset loading command: {dataset_loading_command}"
        )
        dataset_loading_command_exit = run_subprocess(dataset_loading_command)

        if dataset_loading_command_exit != 0:
            raise RuntimeError("Dataset loading failed.")

    config.data.rebuild_dataset_cache = (
        False  # dataset loading is done in the previous step
    )

    yaml_config = OmegaConf.to_yaml(config, resolve=True)
    # TODO(anferico): If the user requested to rebuild the dataset
    # cache, it'll be rebuilt once and for all by the command above, and
    # it should not be rebuilt again by the training command, so we
    # overwrite the value in the config. Note that this way of doing it
    # is utterly horrible, but since we can't modify the config object,
    # we're doing this instead. This is also bad because in the config
    # that's saved on disk, it'll appear that the dataset cache was not
    # rebuilt even if it was
    yaml_config = yaml_config.replace(
        "rebuild_dataset_cache: true", "rebuild_dataset_cache: false"
    )
    config_save_path = Path(config.training.output_dir, "config_hydra.yaml")
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config_save_path.write_text(yaml_config)

    # fmt: off
    train_command = [
        "accelerate", "launch",
            "--config-file", config.accelerate_config,
            "--deepspeed-config-file", config.deepspeed_config,
            "--num-processes", str(config.num_nodes * config.num_gpus),
            "--num-machines", str(config.num_nodes),
            "--machine-rank", os.getenv("SLURM_PROCID", "0"),
            "--main-process-ip", os.getenv("MASTER_ADDR", "localhost"),
            "--main-process-port", os.getenv("MASTER_PORT", "29500"),
            "speechlmm/train/train.py",
            "--config", str(config_save_path),
    ]
    # fmt: on
    logger.info(f"Launching train command: {train_command}")

    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_WATCH"] = config.wandb_watch
    run_subprocess(train_command)


def wire_multimodal_adapters(config: DictConfig) -> DictConfig:
    """Sets input and output dimensions for each of the adapters, as well as
    other configuration options that depend upon the specific encoder and
    decoder it is connected to (e.g. `ctc_loss_vocab_size`)."""

    supported_modalities = ["vision", "audio", "video"]
    for modality in supported_modalities:
        adapter_config = getattr(config.model, f"{modality}_adapter", None)

        if adapter_config is not None:
            encoder_config = getattr(config.model, f"{modality}_encoder")
            text_decoder_config = config.model.text_decoder
            adapter_config.input_dim = encoder_config.hidden_size
            adapter_config.output_dim = text_decoder_config.hidden_size
            adapter_config = _fill_in_missing_parameters(
                adapter_config,
                encoder_config=encoder_config,
                text_decoder_config=text_decoder_config,
            )
            setattr(config.model, f"{modality}_adapter", adapter_config)

        if modality == "audio" and hasattr(config.model, "talking_head"):
            text_decoder_config = config.model.text_decoder
            talking_head_config = config.model.talking_head

            if config.model.use_audio_encoder_as_codec_encoder:
                if config.model.audio_encoder is None:
                    raise ValueError(
                        "If `use_audio_encoder_as_codec_encoder` is True, "
                        "you must provide an `audio_encoder`."
                    )
                logging.info(
                    "Using the `audio_encoder` as the `codec_encoder`."
                    "Codec encoder configuration will be ignored."
                )
                config.model.codec_encoder = config.model.audio_encoder

            codec_encoder_config = config.model.codec_encoder

            if OmegaConf.is_missing(talking_head_config, "num_quantizers"):
                talking_head_config.num_quantizers = (
                    codec_encoder_config.n_quantizers
                )
                talking_head_config.codebook_size = (
                    codec_encoder_config.codebook_size
                )
            setattr(config.model, "talking_head", talking_head_config)

            if modality == "audio" and hasattr(
                config.model, "backfeeding_audio_adapter"
            ):
                backfeeding_audio_adapter_config = (
                    config.model.backfeeding_audio_adapter
                )
                adapter_config = getattr(
                    config.model.backfeeding_audio_adapter, "audio_adapter"
                )

                if (
                    backfeeding_audio_adapter_config.model_type == "features"
                    and OmegaConf.is_missing(adapter_config, "input_dim")
                ):
                    adapter_config.input_dim = (
                        codec_encoder_config.hidden_size
                    )  # Mimi: 512, Encodec: 128
                    adapter_config.output_dim = (
                        text_decoder_config.hidden_size
                    )  # LLM hidden size: 4096
                if (
                    backfeeding_audio_adapter_config.model_type == "codes"
                    and OmegaConf.is_missing(adapter_config, "output_dim")
                ):
                    adapter_config.input_dim = (
                        codec_encoder_config.codebook_size + 1
                    )  # Mimi: 2048+1, Encodec: 1024+1
                    adapter_config.output_dim = (
                        text_decoder_config.hidden_size
                    )  # LLM hidden size: 4096
                setattr(
                    config.model.backfeeding_audio_adapter,
                    "audio_adapter",
                    adapter_config,
                )

            if modality == "audio" and hasattr(
                config.model, "conditioning_audio_adapter"
            ):
                conditioning_audio_adapter_config = (
                    config.model.conditioning_audio_adapter
                )
                if OmegaConf.is_missing(
                    conditioning_audio_adapter_config, "input_dim"
                ):
                    conditioning_audio_adapter_config.input_dim = (
                        codec_encoder_config.hidden_size
                    )
                if OmegaConf.is_missing(
                    conditioning_audio_adapter_config, "output_dim"
                ):
                    conditioning_audio_adapter_config.output_dim = (
                        text_decoder_config.hidden_size
                    )
                setattr(
                    config.model,
                    f"conditioning_audio_adapter",
                    conditioning_audio_adapter_config,
                )

    return config


def _fill_in_missing_parameters(
    adapter_config: DictConfig,
    encoder_config: DictConfig,
    text_decoder_config: DictConfig,
) -> DictConfig:
    if OmegaConf.is_missing(adapter_config, "ctc_loss_vocab_size"):
        adapter_config.ctc_loss_vocab_size = text_decoder_config.vocab_size
    if OmegaConf.is_missing(adapter_config, "ctc_loss_blank_id"):
        adapter_config.ctc_loss_blank_id = getattr(
            text_decoder_config, "pad_token_id", None
        ) or getattr(text_decoder_config, "eos_token_id")

    length_adapter_config = getattr(adapter_config, "length_adapter", None)
    if length_adapter_config is not None:
        if adapter_config.modality_adapter_before is not None:
            adapter_config.modality_adapter_before.input_dim = (
                adapter_config.input_dim
            )
            adapter_config.modality_adapter_before.output_dim = (
                # We let the `modality_adapter_before` perform a
                # projection if required
                adapter_config.output_dim
                # adapter_config.modality_adapter_before.hidden_size
            )
            adapter_config.modality_adapter_before = (
                _fill_in_missing_parameters(
                    adapter_config.modality_adapter_before,
                    encoder_config=encoder_config,
                    text_decoder_config=text_decoder_config,
                )
            )
            length_adapter_config.input_dim = (
                adapter_config.modality_adapter_before.output_dim
            )
        else:
            length_adapter_config.input_dim = adapter_config.input_dim

        if adapter_config.modality_adapter_after is not None:
            length_adapter_config.output_dim = (
                # Again, we let the `modality_adapter_after` perform a
                # projection if required, letting the `length_adapter`
                # focus on length adaptation only
                length_adapter_config.input_dim
            )
            adapter_config.modality_adapter_after.input_dim = (
                length_adapter_config.output_dim
            )
            adapter_config.modality_adapter_after.output_dim = (
                adapter_config.output_dim
                # text_decoder_config.hidden_size
            )
            adapter_config.modality_adapter_after = (
                _fill_in_missing_parameters(
                    adapter_config.modality_adapter_after,
                    encoder_config=encoder_config,
                    text_decoder_config=text_decoder_config,
                )
            )
        else:
            length_adapter_config.output_dim = text_decoder_config.hidden_size

        length_adapter_config = _fill_in_missing_parameters(
            length_adapter_config,
            encoder_config=encoder_config,
            text_decoder_config=text_decoder_config,
        )
        adapter_config.length_adapter = length_adapter_config

    return adapter_config


def handle_missing_keys_and_multirun_quirks(config: DictConfig) -> DictConfig:
    hydra_config = HydraConfig.get()
    modules_identifiers = []
    if hydra_config.runtime.choices["model/audio_encoder"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/audio_encoder"]
        )
    if hydra_config.runtime.choices["model/audio_adapter"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/audio_adapter"]
        )
    if hydra_config.runtime.choices["model/video_encoder"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/video_encoder"]
        )
    if hydra_config.runtime.choices["model/video_adapter"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/video_adapter"]
        )
    if hydra_config.runtime.choices["model/text_decoder"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/text_decoder"]
        )
    if hydra_config.runtime.choices["model/talking_head"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/talking_head"]
        )
    if hydra_config.runtime.choices["model/conditioning_audio_adapter"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/conditioning_audio_adapter"]
        )
    if hydra_config.runtime.choices["model/backfeeding_audio_adapter"]:
        modules_identifiers.append(
            hydra_config.runtime.choices["model/backfeeding_audio_adapter"]
        )

    modules_identifiers = "-".join(modules_identifiers)
    run_identifier = "-".join(
        [
            config.training_type,
            config.training.modality,
            modules_identifiers,
            hydra_config.runtime.choices.training_setting,
        ]
    )

    missing_keys = OmegaConf.missing_keys(config)

    if "training.run_name" in missing_keys:
        config.training.run_name = run_identifier
        missing_keys.remove("training.run_name")
    elif hydra_config.mode == "MULTIRUN":
        provided_run_name = config.training.run_name
        config.training.run_name = f"{provided_run_name}-{run_identifier}"
        print(
            f"You passed run_name='{provided_run_name}', but since this run "
            f"is part of a multirun, it will be overwritten to "
            f"'{config.training.run_name}' to avoid having several runs with the "
            f"same name."
        )

    training_type_to_outdir_prefix = {
        "pretrain": "speechlmm",
        "finetune": "lora-speechlmm",
    }
    if config.training_type not in training_type_to_outdir_prefix:
        raise ValueError(
            f"Unknown training type: {config.training_type}. Expected "
            f"one of {list(training_type_to_outdir_prefix.keys())}."
        )
    output_dir_prefix = training_type_to_outdir_prefix[config.training_type]
    if "training.output_dir" in missing_keys:
        config.training.output_dir = str(
            Path(
                os.environ["CHECKPOINTS_HOME"],
                config.wandb_project,
                f"{output_dir_prefix}-{config.training.run_name}",
            )
        )
        missing_keys.remove("training.output_dir")
    else:
        output_dir = Path(config.training.output_dir)
        if not output_dir.name.startswith(output_dir_prefix):
            raise ValueError(
                f"Output directory '{output_dir.name}' does not start with "
                f"'{output_dir_prefix}' as expected for training type "
                f"'{config.training_type}'."
            )
        if hydra_config.mode == "MULTIRUN":
            provided_output_dir = config.training.output_dir
            config.training.output_dir = str(
                output_dir.with_name(f"{output_dir.name}-{run_identifier}")
            )
            print(
                f"You passed output_dir='{provided_output_dir}', but since "
                f"this run is part of a multirun, it will be overwritten to "
                f"'{config.training.output_dir}' to prevent multiple jobs from "
                f"writing results to the same directory."
            )

    if config.training_type == "finetune":
        if "training.pretrained_checkpoint" in missing_keys:
            output_dir = Path(config.training.output_dir)
            config.training.pretrained_checkpoint = str(
                output_dir.with_name(
                    output_dir.name.replace(
                        training_type_to_outdir_prefix["finetune"],
                        training_type_to_outdir_prefix["pretrain"],
                        1,  # replace the first occurrence
                    ).replace("finetune", "pretrain", 1)
                )
            )
            missing_keys.remove("training.pretrained_checkpoint")

    if config.num_gpus <= 0:
        if hasattr(hydra_config.launcher, "gres"):  # `sbatch` mode
            # NOTE: we cannot determine the number of GPUs from within
            # the login node using `torch.cuda.device_count()`, because
            # there are no GPUs in the login node
            match = re.search(r"gpu:(\d+)", hydra_config.launcher.gres)
            if match is None:
                raise ValueError(
                    "Could not determine the number of GPUs from #SBATCH "
                    "directives. Are you missing a 'gres=gpu:X' directive?"
                )
            config.num_gpus = int(match.group(1))
        else:  # `srun` mode / local run
            import torch  # fmt: skip
            config.num_gpus = torch.cuda.device_count()

    if config.num_nodes <= 0:
        if hasattr(hydra_config.launcher, "nodes"):  # `sbatch` mode
            config.num_nodes = int(hydra_config.launcher.nodes)
        else:  # `srun` mode / local run
            config.num_nodes = 1

    # TODO(anferico): do we want to support different sampling rates for
    # audio encoder and codec decoder?
    if config.data.audio_input_sampling_rate is None:
        if hasattr(config.model, "audio_encoder"):
            audio_input_sampling_rate = (
                config.model.audio_encoder.sampling_rate
            )
            config.data.audio_input_sampling_rate = audio_input_sampling_rate

    if config.data.codec_sampling_rate is None:
        if hasattr(config.model, "codec_encoder"):
            codec_sampling_rate = config.model.codec_encoder.sampling_rate
            config.data.codec_sampling_rate = codec_sampling_rate

    if config.data.codec_frame_rate is None:
        if hasattr(config.model, "codec_encoder"):
            codec_frame_rate = config.model.codec_encoder.frame_rate
            config.data.codec_frame_rate = codec_frame_rate

    if len(missing_keys) > 0:
        raise RuntimeError(f"Missing required keys in config: {missing_keys}")

    return config


def handle_adjustments(config: DictConfig) -> DictConfig:
    if not hasattr(config, "adjustments"):
        return config

    def remove_prefix(string, prefix):
        return string[len(prefix) :] if string.startswith(prefix) else string

    config_hydra = HydraConfig.get()
    for adjustment_name in config.adjustments:
        config_adjustments = hydra.compose(
            config_name=f"adjustment/{adjustment_name}"
        )
        for adjustment in config_adjustments.adjustment.adjustments:
            conditions_met = True
            for (
                attribute,
                required_value_or_values,
            ) in adjustment.conditions.items():
                if attribute.startswith("hydra."):
                    # The attribute value is in the hydra config
                    attribute = remove_prefix(attribute, "hydra.")
                    actual_value = attrgetter(attribute)(config_hydra)
                else:
                    # The attribute value is in the main config
                    actual_value = attrgetter(attribute)(config)

                # NOTE: the conditions in the list are ANDed together.
                # If we wanted to OR them, we would have to add an
                # additional adjustment with different conditions and
                # same overrides
                if not isinstance(required_value_or_values, ListConfig):
                    required_value_or_values = [required_value_or_values]
                # NOTE: if `required_value_or_values` is a list, we
                # check if actual_value=value1 OR actual_value=value2
                # OR ... (it wouldn't make sense to do AND here since a
                # given attribute can't have multiple values at the same
                # time)
                if actual_value not in required_value_or_values:
                    conditions_met = False
                    break
            if conditions_met:
                config = OmegaConf.merge(config, adjustment.overrides)

    return config


def run_subprocess(command: List[str]) -> None:
    # start the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # read process output in real-time
    while process.poll() is None:  # process hasn't finished yet
        output = process.stdout.readline()
        if output:
            print(output.strip("\n"), file=sys.stdout)

    # capture any remaining output after the process has finished
    for output in process.stdout:
        print(output.strip("\n"), file=sys.stdout)

    # return the exit code
    return process.returncode


if __name__ == "__main__":
    train()
