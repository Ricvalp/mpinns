from absl import app, logging
from ml_collections import config_flags
import wandb
from pathlib import Path

from chart_autoencoder import Trainer
from datasets import get_dataset
from chart_autoencoder import (
    plot_3d_charts,
    plot_3d_boundaries,
    plot_local_charts_2d,
    plot_local_charts_2d_with_boundaries,
    plot_html_3d_point_cloud,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    prefetch_to_device,
    ChartSampler,
    get_charts,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "fit", default="fit/config/fit_propeller.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.profiler.log_dir).mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
        )

    train_data = get_dataset(cfg.dataset)

    plot_html_3d_point_cloud(
        points=train_data.data,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_point_cloud.html",
    )

    charts, boundaries, boundary_indices, sampled_points = get_charts(
        points=train_data.data,
        connectivity=train_data.connectivity,
        cfg=cfg,
    )

    data_loader = iter(
        ChartSampler(
            charts=charts,
            batch_size=cfg.train.batch_size,
            distances_matrices_path=cfg.dataset.distances_matrices_path,
            use_existing_distances_matrix=cfg.dataset.use_existing_distances_matrix,
        )
    )

    plot_html_3d_charts(
        charts=charts,
        sampled_points=sampled_points,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.html",
    )
    plot_html_3d_boundaries(
        boundaries=boundaries,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_boundaries.html",
    )
    plot_3d_charts(
        charts=charts,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.png",
    )

    logging.info(f"Number of charts: {len(charts)}")

    trainer = Trainer(
        cfg=cfg,
        data_loader=data_loader,
        charts=charts,
        boundaries=boundaries,
    )

    logging.info("Starting fitting")

    trainer.fit(num_steps=cfg.train.num_steps)
    trainer.load_model(step=cfg.train.num_steps)

    latent_charts = []
    for ckey in charts.keys():
        latent_charts.append(trainer.encoder_fn(charts[ckey], ckey))

    recon_charts = []
    for ckey in charts.keys():
        recon_charts.append(trainer.autoencoder_fn(charts[ckey], ckey))

    plot_html_3d_charts(
        charts=recon_charts,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_charts_3d.html",
        sampled_points=sampled_points,
    )
    plot_3d_charts(
        charts=recon_charts,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_charts_3d.png",
    )
    plot_local_charts_2d(
        charts=latent_charts,
        boundaries_indices=boundary_indices,
        name=Path(cfg.figure_path) / f"post_{cfg.dataset.name}_latent_charts.png",
        s=0.5,
    )

    if cfg.plot.plot_boundaries:
        for bkey in boundaries.keys():

            boundary = boundaries[bkey]
            recon_boundaries = []
            recon_charts = []
            for idx in bkey:
                recon_boundaries.append(trainer.encoder_fn(boundary, idx))
                recon_charts.append(trainer.encoder_fn(charts[idx], idx))

            plot_local_charts_2d_with_boundaries(
                charts=recon_charts,
                boundaries=recon_boundaries,
                name=Path(cfg.figure_path)
                / f"post_{cfg.dataset.name}_boundary_{bkey}.png",
            )


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


if __name__ == "__main__":
    app.run(main)
