PYTHONPATH=. \
python fit/fit_autodecoder.py \
--config=fit/config/fit_autodecoder_sphere.py \
--config.train.reg_lambda=1. \
--config.train.warmup_steps=20000 \
--config.train.lambda_geo_loss=50.