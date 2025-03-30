PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/propeller.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/propeller/checkpoints/best/uibpm3he" \
    --config.eval.step=329999 \
    --config.eval.use_existing_solution=True\
    --config.mode=eval