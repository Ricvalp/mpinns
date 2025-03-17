PYTHONPATH=. NETWORKX_AUTOMATIC_BACKENDS="networkx" JAX_DISABLE_JIT="False" \
    python fit/refine_autodecoder.py \
    --fit=fit/config/fit_bunny_autodecoder.py \
    --fit.wandb.use=False \
    --fit.dataset.load_existing_charts=True \
    --fit.dataset.save_charts=True \
    --fit.charts_to_refine.chart_to_refine=7 \
    --fit.charts_to_refine.min_dist=0.2 \
    --fit.charts_to_refine.nearest_neighbors=10
