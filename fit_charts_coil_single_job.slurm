#!/bin/bash
#SBATCH --job-name=fit_charts_coil
#SBATCH --output=coil_%j.out
#SBATCH --error=coil_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=00:20:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. \
python fit/fit_autodecoder.py \
--config=fit/config/fit_autodecoder_coil.py \
--config.charts_to_fit='(55)' \
--config.train.reg_lambda=2. \
--config.train.warmup_steps=1000 