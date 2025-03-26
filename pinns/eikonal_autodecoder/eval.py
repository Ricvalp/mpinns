from pathlib import Path
import logging

import jax.numpy as jnp
import ml_collections
import models
from tqdm import tqdm

from chart_autoencoder import (
    get_metric_tensor_and_sqrt_det_g_autodecoder,
    load_charts,
    find_intersection_indices,
)

from pinns.eikonal_autodecoder.get_dataset import get_dataset, get_eikonal_gt_solution
from pinns.eikonal_autodecoder.utils import get_last_checkpoint_dir

from jaxpi.utils import restore_checkpoint, load_config
from jaxpi.solution import get_final_solution, load_solution, save_solution

from plot import (
    plot_3d_level_curves,
    plot_3d_solution,
    plot_charts_solution,
    plot_correlation,
)

import jax


def evaluate(config: ml_collections.ConfigDict):

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.eval.solution_path).mkdir(parents=True, exist_ok=True)

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
        N=config.eval.N,
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

    if config.eval.eval_with_last_ckpt:
        last_ckpt_dir = get_last_checkpoint_dir(config.eval.checkpoint_dir)
        ckpt_path = (Path(config.eval.checkpoint_dir) / Path(last_ckpt_dir)).resolve()
    else:
        ckpt_path = Path(config.eval.checkpoint_dir).resolve()

    charts, charts_idxs, boundaries, boundary_indices, charts2d = load_charts(
        charts_path=charts_config.dataset.charts_path,
        from_autodecoder=True,
    )

    eval_name = config.eval.checkpoint_dir.split("/")[-1]

    if config.eval.use_existing_solution:
        pts, sol, u_preds = load_solution(
            config.eval.solution_path + f"/eikonal_solution_{eval_name}.npy"
        )

    else:

        model.state = restore_checkpoint(model.state, ckpt_path, step=config.eval.step)
        params = model.state.params

        u_preds = []

        logging.info(f"Evaluating the solution on the charts")
        for i in tqdm(range(len(x))):
            u_preds.append(
                model.u_pred_fn(jax.tree.map(lambda x: x[i], params), x[i], y[i])
            )
        
        pts, sol = get_final_solution(
            charts=charts,
            charts_idxs=charts_idxs,
            u_preds=u_preds,
        )

        save_solution(
            config.eval.solution_path + f"/eikonal_solution_{eval_name}.npy",
            pts,
            sol,
            u_preds,
        )

    plot_charts_solution(x, y, u_preds, name=config.figure_path + f"/eikonal.png")

    for angles in [(30, 45), (30, 135), (30, 225), (30, 315)]:
        plot_3d_solution(
            pts, sol, angles, config.figure_path + f"/eikonal_3d_{angles[1]}.png"
        )

    for tol in [1e-2, 5e-2, 1e-1, 5e-1]:
        plot_3d_level_curves(
            pts,
            sol,
            tol,
            name=config.figure_path + f"/eikonal_3d_level_curves_{tol}.png",
        )

    mesh_pts, gt_sol = get_eikonal_gt_solution(
        mesh_path=config.mesh.path, scale=config.mesh.scale
    )

    gt_sol_pts_idxs = find_intersection_indices(
        mesh_pts,
        pts,
    )

    assert len(gt_sol_pts_idxs) == len(
        mesh_pts
    ), "The number of points in the mesh and the number of intersection points don't match. Probably due to numerical errors."

    mesh_sol = sol[gt_sol_pts_idxs]

    plot_correlation(
        mesh_sol, gt_sol, name=config.figure_path + f"/eikonal_correlation.png"
    )
