import os
from pathlib import Path
import numpy as np

import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import models
from tqdm import tqdm

from chart_autoencoder import get_metric_tensor_and_sqrt_det_g_autodecoder, load_charts

from pinns.eikonal_autodecoder.get_dataset import get_dataset
from pinns.eikonal_autodecoder.utils import get_last_checkpoint_dir

from jaxpi.utils import restore_checkpoint, load_config

import jax


def evaluate(config: ml_collections.ConfigDict):

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    charts_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), d_params = get_metric_tensor_and_sqrt_det_g_autodecoder(
        charts_config,
        step=config.autoencoder_checkpoint.step,
        inverse=True,
    )

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=charts_config.dataset.charts_path,
        mesh_path=config.mesh.path,
        scale=config.mesh.scale,
        N=config.N
    )

    # Initialize model
    model = models.Diffusion(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        bcs_charts=jnp.array(list(bcs.keys())),
        boundaries=(boundaries_x, boundaries_y),
        num_charts=len(x),
    )

    # Restore the last checkpoint
    if config.eval.eval_with_last_ckpt:
        last_ckpt_dir = get_last_checkpoint_dir(config.eval.checkpoint_dir)
        ckpt_path = (Path(config.eval.checkpoint_dir) / Path(last_ckpt_dir)).resolve()
    else:
        ckpt_path = Path(config.eval.eval_checkpoint_dir).resolve()

    model.state = restore_checkpoint(model.state, ckpt_path, step=config.eval.step)
    params = model.state.params

    u_preds = []

    for i in tqdm(range(len(x))):
        u_preds.append(
            model.u_pred_fn(jax.tree.map(lambda x: x[i], params), x[i], y[i])
        )

    d_params = [jax.tree.map(lambda x: x[i], d_params) for i in range(len(x))]

    vmin = min(np.min(u_pred) for u_pred in u_preds)
    vmax = max(np.max(u_pred) for u_pred in u_preds)

    num_charts = len(x)
    num_rows = int(np.ceil(np.sqrt(num_charts)))
    num_cols = int(np.ceil(num_charts / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, (ax, u_pred_chart) in enumerate(zip(axes, u_preds)):
 
        ax.set_title(f"Chart {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        scatter = ax.scatter(x[i], y[i], c=u_pred_chart, cmap="jet", s=2.5, vmin=vmin, vmax=vmax)
        fig.colorbar(scatter, ax=ax, shrink=0.6)
        
    plt.tight_layout()
    plt.savefig(config.figure_path + f"/eikonal.png")
    plt.close()

    for angles in [(30, 45), (30, 135), (30, 225), (30, 315)]:

        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        
        scatter = None
        for i in range(len(u_preds)):
            u = u_preds[i]
            X = decoder.apply(
                {"params": d_params[i]}, jnp.stack([x[i], y[i]], axis=1)
            )
            scatter = ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=u,
                cmap="jet",
                s=2.5,
                vmin=vmin,
                vmax=vmax,
            )

            ax.view_init(angles[0], angles[1])
        
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label('u value')
        
        plt.tight_layout()
        plt.savefig(config.figure_path + f"/eikonal{angles[1]}.png")
        plt.close()
        
    
        charts, boundaries, boundary_indices, charts2d = load_charts(
            charts_path=charts_config.dataset.charts_path,
            from_autodecoder=True,
        )


    
    sol = get_final_solution(
        charts,
        boundaries,
        boundary_indices,
        u_preds,
    )

        

    for angles in [(30, 45), (30, 135), (30, 225), (30, 315)]:

        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        
        scatter = None
        for i in range(len(u_preds)):
            u = u_preds[i]
            X = decoder.apply(
                {"params": d_params[i]}, jnp.stack([x[i], y[i]], axis=1)
            )
            scatter = ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=u,
                cmap="jet",
                s=2.5,
                vmin=vmin,
                vmax=vmax,
            )

            ax.view_init(angles[0], angles[1])
        
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label('u value')
        
        plt.tight_layout()
        plt.savefig(config.figure_path + f"/eikonal{angles[1]}.png")
        plt.close()