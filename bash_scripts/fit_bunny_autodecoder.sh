PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" \
    python fit/fit_autodecoder.py \
    --fit=fit/config/fit_bunny_autodecoder.py \
    --fit.wandb.use=True \
    --fit.dataset.load_existing_charts=True \
    --fit.umap.use_existing_umap_embeddings=True \
    --fit.dataset.use_existing_distances_matrix=True \
    --fit.checkpoint.checkpoint_path="./fit/checkpoints/bunny" \
