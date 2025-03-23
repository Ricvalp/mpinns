from pathlib import Path

import ml_collections
import models
from tqdm import tqdm
from samplers import (
    UniformBCSampler,
    UniformSampler,
    UniformBoundarySampler,
)

import jax.numpy as jnp

from chart_autoencoder.riemann import get_metric_tensor_and_sqrt_det_g_autodecoder

from pinns.eikonal_autodecoder.get_dataset import get_dataset

from pinns.eikonal_autodecoder.plot import (
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
    wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        entity=wandb_config.entity,
        config=config,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.profiler.log_dir).mkdir(parents=True, exist_ok=True)
    logger = Logger()

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), d_params = get_metric_tensor_and_sqrt_det_g_autodecoder(
        autoencoder_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=autoencoder_config.dataset.charts_path,
        mesh_path=config.mesh.path,
        scale=config.mesh.scale,
        N=config.N,
    )

    if config.plot:

        plot_domains(
            x,
            y,
            boundaries_x,
            boundaries_y,
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            name=Path(config.figure_path) / "domains.png",
        )

        plot_domains_3d(
            x,
            y,
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
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

    bcs_sampler = iter(
        UniformBCSampler(
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            num_charts=len(x),
            batch_size=config.training.batch_size,
            bcs_batches_path=(
                config.training.bcs_batches_path,
                config.training.bcs_values_path,
            ),
            load_existing_batches=config.training.load_existing_batches,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=0.1,
            batch_size=config.training.batch_size,
        )
    )

    boundary_sampler = iter(
        UniformBoundarySampler(
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            batch_size=config.training.batch_size,
            boundary_batches_paths=(
                config.training.boundary_batches_path,
                config.training.boundary_pairs_idxs_path,
            ),
            load_existing_batches=config.training.load_existing_batches,
        )
    )

    model = models.Eikonal(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        bcs_charts=jnp.array(list(bcs.keys())),
        boundaries=(boundaries_x, boundaries_y),
        num_charts=len(x),
    )

    evaluator = models.EikonalEvaluator(config, model)

    print("Waiting for JIT...")

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(boundary_sampler), next(bcs_sampler)
        loss, model.state = model.step(model.state, batch)

        if step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss}, step)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

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

    # for step in tqdm(range(step, step + config.training.lbfgs_max_steps + 1), desc="L-BFGS"):

    #     # set_profiler(config.profiler, step, config.profiler.log_dir)

    #     batch = next(res_sampler), next(boundary_sampler), next(ics_sampler)
    #     loss, model.state = model.lbfgs_step(model.state, batch)

    #     if step % config.wandb.log_every_steps == 0:
    #         wandb.log({"loss": loss}, step)

    return model
