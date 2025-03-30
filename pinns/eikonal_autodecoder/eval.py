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
    model_config = load_config(
        Path(config.eval.checkpoint_dir) / "cfg.json",
    )
    
    eval_config = config.eval

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
        N=eval_config.N,
    )
    


    model = models.Eikonal(
        model_config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        d_params=d_params,
        bcs_charts=jnp.array(list(bcs.keys())),
        boundaries=(boundaries_x, boundaries_y),
        num_charts=len(x),
    )

    if eval_config.eval_with_last_ckpt:
        last_ckpt_dir = get_last_checkpoint_dir(eval_config.checkpoint_dir)
        ckpt_path = (Path(eval_config.checkpoint_dir) / Path(last_ckpt_dir)).resolve()
    else:
        ckpt_path = Path(eval_config.checkpoint_dir).resolve()

    charts, charts_idxs, boundaries, boundary_indices, charts2d = load_charts(
        charts_path=charts_config.dataset.charts_path,
        from_autodecoder=True,
    )

    eval_name = eval_config.checkpoint_dir.split("/")[-1]

    if eval_config.use_existing_solution:
        pts, sol, u_preds = load_solution(
            eval_config.solution_path + f"/eikonal_solution_{eval_name}.npy"
        )

    else:

        model.state = restore_checkpoint(model.state, ckpt_path, step=eval_config.step)
        params = model.state.params

        u_preds = []
        
        u_pred_fn = jax.jit(
            model.u_pred_fn
            )

        logging.info("Evaluating the solution on the charts")
        for i in tqdm(range(len(x))):
            u_preds.append(
                model.u_pred_fn(jax.tree.map(lambda x: x[i], params), x[i], y[i])
            )
        
        logging.info("Joining solutions")
        pts, sol = get_final_solution(
            charts=charts,
            charts_idxs=charts_idxs,
            u_preds=u_preds,
        )

        save_solution(
            eval_config.solution_path + f"/eikonal_solution_{eval_name}.npy",
            pts,
            sol,
            u_preds,
        )

    plot_charts_solution(x, y, u_preds, name=config.figure_path + "/eikonal.png")

    for angles in [(30, 45)]: #, (30, 135), (30, 225), (30, 315)]:
        plot_3d_solution(
            pts, sol, angles, config.figure_path + f"/eikonal_3d_{angles[1]}.png", s=2.5,
        )

    # for tol in [1e-2, 5e-2, 1e-1, 5e-1]:
    #     plot_3d_level_curves(
    #         pts,
    #         sol,
    #         tol,
    #         name=config.figure_path + f"/eikonal_3d_level_curves_{tol}.png",
    #     )

    mesh_pts, gt_sol = get_eikonal_gt_solution(
        charts_path=charts_config.dataset.charts_path,
    )

    gt_sol_pts_idxs = find_intersection_indices(
        mesh_pts,
        pts,
    )

    for angles in [(30, 45)]: #, (30, 135), (30, 225), (30, 315)]:
        plot_3d_solution(
            mesh_pts, gt_sol, angles, config.figure_path + f"/gt_eikonal_3d_{angles[1]}.png", s=15,
        )

    assert len(gt_sol_pts_idxs) == len(
        mesh_pts
    ), "The number of points in the mesh and the number of intersection points don't match. Probably due to numerical errors."

    mesh_sol = sol[gt_sol_pts_idxs]

    MSE = jnp.mean(((mesh_sol - gt_sol)/mesh_sol.mean()) ** 2)
    print(f"MSE: {MSE}")
    print(f"Correlation: {jnp.corrcoef(mesh_sol, gt_sol)[0, 1]}")
    plot_correlation(
        mesh_sol, gt_sol, name=config.figure_path + "/eikonal_correlation.png"
    )
