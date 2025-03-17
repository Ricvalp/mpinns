import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_collections
import models
from tqdm import tqdm
from jax.tree_util import tree_map
from samplers import (
    ICUniformSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from chart_autoencoder.riemann import (
    get_metric_tensor_and_sqrt_det_g,
    get_metric_tensor_and_sqrt_det_g_autodecoder,
)
from chart_autoencoder.utils import prefetch_to_device

from pinns.diffusion_single_gpu.get_dataset import get_dataset

from pinns.diffusion_single_gpu.plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
)

import wandb
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, load_config

from utils import set_profiler

import matplotlib.pyplot as plt
import numpy as np


def train_and_evaluate(config: ml_collections.ConfigDict):

    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.profiler.log_dir).mkdir(parents=True, exist_ok=True)
    logger = Logger()

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    (
        inv_metric_tensor,
        sqrt_det_g,
        encoder,
        decoder,
    ), (e_params, d_params) = get_metric_tensor_and_sqrt_det_g(
        autoencoder_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, u0, boundaries_x, boundaries_y = get_dataset(
        encoder,
        e_params,
        autoencoder_config.dataset.charts_path,
        sigma=0.1,
    )

    plot_domains(
        x,
        y,
        boundaries_x,
        boundaries_y,
        ics=u0,
        name=Path(config.figure_path) / "domains.png",
    )

    if config.plot:
        plot_domains_3d(
            x,
            y,
            ics=u0,
            decoder=decoder,
            d_params=d_params,
            name=Path(config.figure_path) / "domains_3d.png",
        )

        plot_domains_with_metric(
            x,
            y,
            sqrt_det_g,
            d_params=d_params,
            name=Path(config.figure_path) / "domains_with_metric.png",
        )

        plot_combined_3d_with_metric(
            x,
            y,
            decoder=decoder,
            sqrt_det_g=sqrt_det_g,
            d_params=d_params,
            name=Path(config.figure_path) / "combined_3d_with_metric.png",
        )

    # ics_sampler = prefetch_to_device(
    #     iter(
    #         ICUniformSampler(
    #             x=x,
    #             y=y,
    #             u0=u0,
    #         )
    #     ),
    #     size=8,
    #     devices=jax.devices(),
    # )

    # res_sampler = prefetch_to_device(
    #     iter(
    #         UniformSampler(
    #             x=x,
    #             y=y,
    #             sigma=0.1,
    #             T=config.T,
    #             batch_size=config.training.batch_size,
    #         )
    #     ),
    #     size=8,
    #     devices=jax.devices(),
    # )

    # boundary_sampler = prefetch_to_device(
    #     iter(
    #         UniformBoundarySampler(
    #             boundaries_x=boundaries_x,
    #             boundaries_y=boundaries_y,
    #             T=config.T,
    #             batch_size=config.training.batch_size,
    #         )
    #     ),
    #     size=8,
    #     devices=jax.devices(),
    # )

    ics_sampler = iter(
        ICUniformSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=0.1,
            T=config.T,
            batch_size=config.training.batch_size,
        )
    )

    boundary_sampler = iter(
        UniformBoundarySampler(
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            T=config.T,
            batch_size=config.training.batch_size,
        )
    )

    model = models.Diffusion(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        ics=(x, y, u0),
        boundaries=(boundaries_x, boundaries_y),
    )

    evaluator = models.DiffusionEvaluator(config, model)

    print("Waiting for JIT...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(boundary_sampler), next(ics_sampler)
        loss, model.state = model.step(model.state, batch)

        wandb.log({"loss": loss}, step)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # if step % config.logging.log_every_steps == 0:
        #     # Get the first replica of the state and batch
        #     state = tree_map(lambda x: x[0], model.state)
        #     batch = tree_map(lambda x: x[0], batch)
        #     log_dict = evaluator(state, batch, None)  # u_ref)
        #     wandb.log(log_dict, step)

        #     logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_checkpoint(
                    model.state,
                    config.saving.checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

    return model
