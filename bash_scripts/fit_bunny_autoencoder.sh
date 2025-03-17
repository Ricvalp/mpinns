PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" \
    python fit/fit_autoencoder.py \
    --fit=fit/config/fit_bunny_autoencoder.py \
    --fit.wandb.use=True \
