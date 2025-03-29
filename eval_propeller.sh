PYTHONPATH=. \
    python pinns/eikonal_autodecoder/main.py \
    --config=pinns/eikonal_autodecoder/configs/propeller.py \
    --config.eval.checkpoint_dir="pinns/eikonal_autodecoder/propeller/checkpoints/best/6jvovynq" \
    --config.eval.step=179999 \
    --config.mode=eval