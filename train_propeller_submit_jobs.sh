#!/bin/bash

rm -rf slurm_logs_train
mkdir -p slurm_logs_train

NUM_LAYERS=(2 4)
LEARNING_RATES=(0.001 0.0001)
DECAY_STEPS=(10000 40000)
ACTIVATION_FUNCTIONS=("gelu" "tanh" "sin")

job_counter=0
for num_layer in ${NUM_LAYERS[@]}; do
    for lr in ${LEARNING_RATES[@]}; do
        for decay_step in ${DECAY_STEPS[@]}; do
            for activation in ${ACTIVATION_FUNCTIONS[@]}; do
                job_counter=$((job_counter + 1))
                cat << EOF > slurm_logs_train/job_${job_counter}.slurm
#!/bin/bash
#SBATCH --job-name=train_${job_counter}
#SBATCH --output=slurm_logs_train/train_${job_counter}_%j.out
#SBATCH --error=slurm_logs_train/train_${job_counter}_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. \
python pinns/eikonal_autodecoder/main.py \
--config=pinns/eikonal_autodecoder/configs/propeller.py \
--config.autoencoder_checkpoint.step=60000 \
--config.mode=train \
--config.optim.learning_rate=${lr} \
--config.arch.num_layers=${num_layer} \
--config.arch.activation=${activation} \
--config.optim.decay_steps=${decay_step} \

EOF
                sbatch slurm_logs_train/job_${job_counter}.slurm
            done
        done
    done
done


