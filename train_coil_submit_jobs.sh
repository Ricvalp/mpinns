#!/bin/bash

rm -rf slurm_logs_train
mkdir -p slurm_logs_train

NUM_LAYERS=(3 4 5)
LEARNING_RATES=(0.001 0.0001 0.00001)

job_counter=0
for num_layer in ${NUM_LAYERS[@]}; do
    for lr in ${LEARNING_RATES[@]}; do
        job_counter=$((job_counter + 1))
        cat << EOF > slurm_logs_train/job_${job_counter}.slurm
#!/bin/bash
#SBATCH --job-name=train_${job_counter}
#SBATCH --output=slurm_logs_train/train_${job_counter}_%j.out
#SBATCH --error=slurm_logs_train/train_${job_counter}_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=06:00:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. \
python pinns/diffusion_single_gpu_autodecoder/main.py \
--config=pinns/diffusion_single_gpu_autodecoder/configs/coil.py \
--config.autoencoder_checkpoint.step=60000 \
--config.mode=train \
--config.optim.learning_rate=${lr} \
--config.arch.num_layers=${num_layer} \
--config.saving.checkpoint_dir=pinns/diffusion_single_gpu_autodecoder/coil/checkpoints/${num_layer}layers_${lr}lr

EOF

        sbatch slurm_logs_train/job_${job_counter}.slurm
    done
done


