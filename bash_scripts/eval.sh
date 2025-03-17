PYTHONPATH=. \
    python pinns/diffusion_single_gpu/main.py \
    --config=pinns/diffusion_single_gpu/configs/default.py \
    --config.autoencoder_checkpoint.step=50000 \
    --config.mode=eval \
    --config.eval.step=14999

