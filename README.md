> [!NOTE]
> If you're working on a Cyfronet machine, please refer to [README_cyfronet.md](README_cyfronet.md) instead.

# 📹🎤 SpeechLMM
This repository contains the code for the SpeechLMM foundation model developed as part of the [Meetween](https://meetween.eu) project. Across the 4-year timeframe of the project (2024-2027), we will release 3 different generations of the model, each time in 4 different sizes (S, M, L, XL).

Below is an illustration of the architecture of SpeechLMM version 1.0:
![SpeechLMM architecture](/assets/images/speechlmm_architecture.png)

## 📖 Contents
- [Preliminary setup](#preliminary-setup)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Datasets and dataloaders](#datasets-and-dataloaders)
- [Codebase](#codebase)
- [Training](#training)
- [Inference](#inference)
- [Known issues](#known-issues)

## 🛠️ Preliminary setup
- SpeechLMM builds on existing foundation models for the different modalities it supports. Some of these models are hosted on Hugging Face but are _gated_ by default, so you must request access to them before you can use them within SpeechLMM. At the moment, you are required to request access to the following models:
  - [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  - [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

- In order for the codebase to work properly, you need to set the following environment variables:
    ```Shell
    # Directory where your datasets reside
    export DATA_HOME=...
    # Path to this repository
    export SPEECHLMM_ROOT=...
    # Directory where the pre-trained components (e.g. modality encoders) are stored
    export PRETRAINED_COMPONENTS=...
    # Directory where model checkpoints will be stored
    export CHECKPOINTS_HOME=...
    ```
    For convenience, you can add the exports above to your `~/.bashrc` or `~/.zshrc` file, replacing the dots with the actual paths.

- Download pre-trained building blocks for SpeechLMM. Important: you must download these models in `$PRETRAINED_COMPONENTS`
  1. SeamlessM4T v2
     ```python
     import os
     from transformers import AutoProcessor, AutoModel
     
     model_name = "facebook/seamless-m4t-v2-large"
     processor = AutoProcessor.from_pretrained(model_name)
     model = AutoModel.from_pretrained(model_name)
     
     processor.save_pretrained(os.path.join(os.getenv("PRETRAINED_COMPONENTS"), model_name))
     model.speech_encoder.save_pretrained(os.path.join(os.getenv("PRETRAINED_COMPONENTS"), model_name))
     ```
  2. Whisper v3
     ```python
     import os
     from transformers import AutoProcessor, AutoModel
     
     model_name = "openai/whisper-large-v3"
     processor = AutoProcessor.from_pretrained(model_name)
     model = AutoModel.from_pretrained(model_name)
     
     processor.save_pretrained(os.path.join(os.getenv("PRETRAINED_COMPONENTS"), model_name))
     model.encoder.save_pretrained(os.path.join(os.getenv("PRETRAINED_COMPONENTS"), model_name))
     ```
  3. AutoAVSR

     Download the checkpoint manually from [https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i](https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i) and put it in `$PRETRAINED_COMPONENTS/`.

## 📊 System requirements

- The codebase has only been tested on Linux
- We only tested training and inference on NVIDIA Ampere (A100), Hopper (H100) and Grace Hopper (GH200) architectures with ≥40GB of VRAM per GPU

## 🔧 Installation

1. Clone this repository and navigate to the `speechlmm` folder 📁

    ```bash
    git clone https://github.com/Meetween/speechlmm.git
    cd speechlmm
    ```

2. Install package using conda 🐍

    ```Shell
    conda create -n speechlmm python=3.10 -y
    conda activate speechlmm
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    ```

3. Install additional packages for training and development 🏋️

    ```
    pip install -e ".[train,dev]"
    pip install flash-attn --no-build-isolation
    ```

4. [Optional, but strongly encouraged 😇] Install `pre-commit` hooks for automatic code formatting 🪝
    ```
    pre-commit install
    ```

5. Install `decord` (library for decoding videos) 🎥

    ```bash
    pip install decord
    ```

6. Apply patches 🩹

    We require some patches to be applied to some of this package's dependencies. In order to apply them, make sure you activated the virtual environment you set up <u>in step 2</u> and run:
    ```sh
    python apply_patches.py
    ```

### Upgrade to latest code base
```Shell
git pull
```
## ⌨️ CLI Inference
To run inference using a trained model, run the following command (on a GPU instance):

```Shell
python speechlmm/serve/cli.py --model-path /path/to/model_directory
```

While chatting with the model, there are three strings that are not passed directly by the model but are handled differently:
1. if you send an empty message, the script terminates.
2. if you write `<reset>`, you clear the conv history. This also clears the audio tokens from the conversation.
3. if you write `audio:/path/to/new/audio_file` at the end of your message, the model will clear the conv history, load a new audio file

## 💾 Datasets and dataloaders

Dataset specifications are contained inside [conf/datasets](conf/datasets). If you wish to contribute a new dataset, follow the instructions in [docs/custom_datasets.md](docs/custom_datasets.md).

## 👩‍💻 Codebase

The script you should use for training models is [`speechlmm/train/train_hydra.py`](speechlmm/train/train_hydra.py), whereas [`speechlmm/train/eval_hydra.py`](speechlmm/eval/eval_hydra.py) should be used for evaluating trained models. The sections below offer an overview of the different modules of this repository, useful if you wish to contribute changes.

### Model
Most of the code that implements SpeechLMM is found in [`modeling_speechlmm.py`](speechlmm/model/modeling_speechlmm.py). The model class is loosely inspired by Hugging Face `transformers` (and it also has an associated configuration class in [`configuration_speechlmm.py`](speechlmm/model/configuration_speechlmm.py)).

### Multimodal encoders, adapters and decoders
[Multimodal encoders](speechlmm/model/encoders), [multimodal adapters](speechlmm/model/adapters) and [multimodal decoders](speechlmm/model/decoders) are organized into folders. Right now we support the audio and vision modality for encoders and adapters, while decoders support only the text modality. If you wish to contribute additional modalities, make sure to follow the same implementation scheme.

## 🏋🏼‍♀️ Training
Launching a training is as simple as running a command like the following:
```sh
python speechlmm/train/train_hydra.py \
    --config-name pretrain \
    model/audio_encoder=seamless \
    model/audio_adapter=mlp \
    model/text_decoder=llama_3_8b \
    training_setting=paper_1a
```

To specify training configurations, we use [Hydra](https://hydra.cc/), which in turn is based on [OmegaConf](https://omegaconf.readthedocs.io/). In the example above, we are launching a pre-training job using SeamlessM4T v2 as the audio encoder, a simple MLP as the audio adapter, and Llama 3 8B as the text decoder. The "training setting" we are using is `paper_1a`, and the details associated with it (such as which datasets to train on, which tasks, ...) are in `conf/speechlmm/training_setting/paper_1a.yaml`. Note that this file does not contain all the configuration options, but only those that differ from the default ones found in `conf/speechlmm/pretrain.yaml` (hence `--config-name pretrain`).

If you wish to tweak any configuration options, you can do so by creating a new YAML file under `conf/speechlmm/training_setting/` and passing that as the `training_setting` parameter. Alternatively, you can override specific parameters in the command line directly, such as `training.per_device_train_batch_size=16`.

### Fine-tuning
The procedure for fine-tuning a pre-trained model is not very different than the one for pre-training (as shown above). The only important difference is that you *must* specify the `training.pretrained_checkpoint` parameter, which should point to a directory containing a pre-trained model checkpoint (in particular, a directory containing a `config.json` and a `model.safetensors` file).

For example, let's imagine you trained a model using the command above and the final checkpoint was saved in `/path/to/speechlmm-pretrain-paper_1a`. Now you want to fine-tune it on a new dataset, so you create `conf/datasets/my_finetuning_dataset.yml` containing your dataset configuration and `conf/speechlmm/training_setting/my_finetuning.yaml` with the following content:
```yaml
⋮

data:
  ⋮    
  data_config_path: conf/datasets/my_finetuning_dataset.yml
  ⋮

training:
  ⋮
  pretrained_checkpoint: /path/to/speechlmm-pretrain-paper_1a
  ⋮

⋮
```

At this point, you can launch the fine-tuning job with the following command:
```sh
python speechlmm/train/train_hydra.py \
    model/audio_encoder=seamless \
    model/audio_adapter=mlp \
    model/text_decoder=llama_3_8b \
    training_setting=my_finetuning
```

### Parameter-efficient pretraining / fine-tuning using LoRA
If you want to run a parameter-efficient pretraining or fine-tuning using LoRA, you must provide an appropriate value for the `training.lora_adapters` configuration parameter. For example, here's a possible configuration where we apply two different LoRA adapters, one to the text decoder and one to the audio encoder:
```yaml
⋮

training:
  ⋮
  lora_adapters:
    - name: text_decoder_peft_adapter
      target_module: text_decoder.model
      task_type: CAUSAL_LM
      r: 128
      lora_alpha: 256
      lora_dropout: 0.05
      bias: none
      use_rslora: true
    - name: audio_encoder_peft_adapter
      target_module: audio_encoder.model
      task_type: FEATURE_EXTRACTION
      r: 64
      lora_alpha: 128
      lora_dropout: 0.05
      bias: none
      use_rslora: true
  ⋮

⋮
```

### Keeping certain parameters frozen
If you want to keep certain parameters frozen during training, you can do so by setting the `training.freeze_modules` configuration parameter to a list of module names. For example, here's how you could keep the audio encoder and text decoder frozen and train only the adapter between them:
```yaml
⋮

training:
  ⋮
  freeze_modules:
    - audio_encoder
    - text_decoder
  ⋮

⋮
```

### 🧹 Hydra _sweeps_
In order to launch training _sweeps_ covering several hyperparameter configurations in parallel, you can use the following command:
```sh
python speechlmm/train/train_hydra.py \
    --multirun \
    --config-name pretrain \
    hydra/launcher=helios \
    model/audio_encoder=hubert,seamless,whisper_unrestricted \
    model/audio_adapter=mlp,cformer_cif,qformer \
    model/text_decoder=llama_3_8b,mistral \
    training_setting=paper_1a,paper_1b,paper_1c,paper_1d \
    adjustments=[zero3_issue,paper_helios]
```
How to interpret the command above:
1. if `--multirun` is specified and you pass arguments of the form `key=value1,...,valueN`, Hydra will launch a number of separate experiments equal to the number of possible combinations of the specified values. For instance, in the example above we have 3 audio encoders, 3 audio adapters, 2 text decoders and 4 training settings, corresponding to 3\*3\*2\*4 = 72 total experiments. Note that `adjustments` represents an exception: the values are wrapped in square brackets, meaning that each of the 72 jobs will use all the "adjustments" specified there
2. if you wish to submit your jobs to a SLURM cluster like Athena or Helios, you should pass `hydra/launcher=athena` or `hydra/launcher=helios`, respectively. If you want to launch your jobs locally, then you should omit the `hydra/launcher` parameter. In fact, for local setups (i.e. on interactive GPU nodes), it usually makes sense to omit `--multirun` too and provide exactly 1 `value` for each `key` (as we did in the previous example)


### Multi-node trainings
At the moment, multi-node trainings are supported only in "sbatch" mode. In other words, you're required to use the `--multirun` flag, <u>even if you don't want to launch multiple experiments at the same time</u>.

To instruct the training script to launch the job on X nodes, simply pass `hydra.launcher.nodes=X`, for example:
```bash
python3 speechlmm/train/train_hydra.py \
    --multirun \
    hydra/launcher=helios \
    model/video_encoder=auto_avsr \
    model/video_adapter=mlp \
    model/text_decoder=llama_3_8b \
    training_setting=vsr_lrs2 \
    hydra.launcher.nodes=2
```

### Hydra quirks
- Always launch `train_hydra.py` within a `tmux` or `screen` session. This is needed because the training is actually started with a subprocess call to `accelerate launch`, so in order to keep track of stdout and stderr we need to keep the parent process running
- Hydra-related logs are stored in the `outputs` and `multirun` directories. These may store a lot of files over time, so make sure to clean them up if you have disk quota issues
- Fine-tuning requires you to specify the model checkpoint to start training from, which can be a bit problematic to do when you do a multirun. For this reason, that parameter is filled in automatically when you pass `--multirun`, <u>but this only works under the following assumptions</u>:
  1. (reasonable) the model checkpoint to start training from is located in the same parent directory where the output of the fine-tuning will be saved. For example, if the output of the fine-tuning will be stored in `/path/to/lora-speechlmm-finetune-whatever`, then the pre-training checkpoint is assumed to be located in `/path/to/speechlmm-pretrain-whatever`
  2. (not so reasonable) the fine-tuning uses the same training setting as the pre-training. For example, if the output of the fine-tuning will be stored in `/path/to/lora-speechlmm-finetune-whatever-some_setting_name`, then the pre-training checkpoint is assumed to be located in `/path/to/speechlmm-pretrain-whatever-some_setting_name`

## 🎲 Inference

### Online inference (chat)
To run inference in an *online* fashion, run the following command on a GPU instance:

```Shell
python speechlmm/serve/cli.py --model-path /path/to/model_checkpoint
```

While chatting with the model, there are three strings that are not passed directly by the model but are handled differently:
1. if you send an empty message, the script terminates.
2. if you write `<reset>`, you clear the conv history. This also clears the audio tokens from the conversation.
3. if you write `audio:/path/to/new/audio_file` at the end of your message, the model will clear the conv history, load a new audio file

### Offline inference
To run inference in an offline fashion, e.g. to use it for batch predictions, run `scripts/inference.py`. For example, if you want to run ASR inference on a bunch of audios inside a directory located at `/path/to/audios`, run:
```sh
python scripts/inference.py \
    --model-path /path/to/model_checkpoint \
    --task asr \
    --audios-file-path /path/to/audios
```
Alternatively, if you want to run inference on the test split of a specific dataset for which you have a configuration file, run:
```sh
python scripts/inference.py \
    --model-path /path/to/model_checkpoint \
    --task asr \
    --dataset_config conf/datasets/my_dataset.yml
```

Note that at the moment, only the following tasks are supported: `asr`, `tts`, `s2st`, `s2st_cloning`. Furthermore, certain tasks (such as `s2st_cloning`) require additional parameters (e.g. `--source-language` and `--target-language`). For more details, run `python scripts/inference.py --help`.

## ⚠️ Known issues
- For some reason, **HuBERT** is not compatible with DeepSpeed ZeRO-3 (the training simply hangs while running `HubertModel.from_pretrained(...)`). If you want to use HuBERT as the audio encoder, you must use ZeRO-0, ZeRO-1 or ZeRO-2

## 🙏 Acknowledgements
This codebase started as a fork of [haotian-liu/LLaVA](https://github.com/haotian-NOTE/speechlmm), although it has been almost completely rewritten.
