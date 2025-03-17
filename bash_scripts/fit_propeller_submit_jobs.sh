#!/bin/bash

rm -rf slurm_logs_propeller
mkdir -p slurm_logs_propeller

REG_LAMBDAS=("0.0005")
REG_TYPES=("reg+iso")
MIN_DIST=("0.5")
N_HIDDENS=("128")

job_counter=0
for reg_lambda in ${REG_LAMBDAS[@]}; do
    for reg_type in ${REG_TYPES[@]}; do
        for min_dist in ${MIN_DIST[@]}; do
            for n_hidden in ${N_HIDDENS[@]}; do
                job_counter=$((job_counter + 1))
                cat << EOF > slurm_logs_propeller/job_${job_counter}.slurm
#!/bin/bash
#SBATCH --job-name=propeller_${job_counter}
#SBATCH --output=slurm_logs_propeller/propeller_${job_counter}_%j.out
#SBATCH --error=slurm_logs_propeller/propeller_${job_counter}_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --gpus=1

source ~/.bashrc
conda init
conda activate manifold-pinns

PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" python fit/fit.py \
    --fit=fit/config/fit_propeller.py \
    --fit.wandb.use=False \
    --fit.dataset.load_existing_charts=False \
    --fit.dataset.use_existing_distances_matrix=False \
    --fit.charts.alg=fast_region_growing \
    --fit.dataset.save_charts=True \
    --fit.train.num_steps=1000000 \
    --fit.checkpoint.save_every=20000 \
    --fit.dataset.name=Propeller \
    --fit.dataset.subset_cardinality=100000 \
    --fit.dataset.scale=0.02 \
    --fit.train.reg_lambda=${reg_lambda} \
    --fit.charts.min_dist=${min_dist} \
    --fit.train.reg=${reg_type} \
    --fit.model.n_hidden=${n_hidden} \
    --fit.dataset.charts_path=datasets/propeller/charts/${reg_lambda}_${reg_type}_${min_dist}_${n_hidden} \
    --fit.checkpoint.checkpoint_path=fit/checkpoints/propeller/${reg_lambda}_${reg_type}_${min_dist}_${n_hidden} \
EOF

                sbatch slurm_logs_propeller/job_${job_counter}.slurm
            done
        done
    done
done


