PYTHONPATH=. \
    python pinns/diffusion_single_gpu_autodecoder/main.py \
    --config=pinns/diffusion_single_gpu_autodecoder/configs/coil.py \
    --config.autoencoder_checkpoint.step=60000 \
    --config.mode=eval 

