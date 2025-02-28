#!/bin/bash

mkdir -p $SCRATCH/conda_pkgs_dir
mkdir -p $SCRATCH/pip_pkgs_dir
mkdir -p $SCRATCH/envs
mkdir -p $SCRATCH/.cache/

echo "export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs_dir" >> ~/.bashrc
echo "export CONDA_ENVS_DIRS=$SCRATCH/envs" >> ~/.bashrc
echo "export PIP_CACHE_DIR=$SCRATCH/pip_pkgs_dir" >> ~/.bashrc
echo "export HF_HOME=$SCRATCH/.cache/huggingface" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache/huggingface/hub" >> ~/.bashrc
echo "export RAY_TMPDIR=$SCRATCH/ray_temp" >> ~/.bashrc

module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"

source ~/.bashrc

if ! [ -d $SCRATCH/envs/speechlmm ]; then
    echo "creating conda environment"
    conda create -p $SCRATCH/envs/speechlmm python==3.10 -y
fi


conda activate speechlmm

echo "installing packages"
pip install --upgrade pip
pip install -e .

# if script was executed with --train
case "$1" in
    --train)
    echo "installing training packages"
    pip install -e ".[train,dev]"
    pip install flash-attn --no-build-isolation
    ;;
esac
