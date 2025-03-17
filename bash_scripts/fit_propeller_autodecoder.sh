PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" \
    python fit/fit_autodecoder.py \
    --fit=fit/config/fit_propeller_autodecoder.py \
    --fit.wandb.use=True \
    --fit.dataset.load_existing_charts=True \
    --fit.charts.alg=fast_region_growing \
    --fit.dataset.save_charts=False \
    --fit.train.num_steps=20001 \
    --fit.train.reg_lambda=0.1 \
    --fit.dataset.name=Propeller \
    --fit.charts.min_dist=0.6 \
    --fit.train.reg=reg+geo \
    --fit.dataset.use_existing_distances_matrix=True \
    --fit.dataset.save_distances_matrix=False \
    --fit.dataset.distances_matrices_path=./datasets/propeller/charts/distances_matrices.npy \
    --fit.umap.use_existing_umap_embeddings=True \
    --fit.umap.umap_embeddings_path=./datasets/propeller/charts/umap_embeddings.npy