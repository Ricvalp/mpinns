from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Charts to fit  # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts_to_fit = (0, 1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Wandb  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "chart-autodecoder"
    cfg.wandb.name = "sphere"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_every_steps = 100
    cfg.wandb.log_charts_every = 30000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Profiler # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.profiler = ConfigDict()
    cfg.profiler.start_step = 1000000
    cfg.profiler.end_step = 1000000
    cfg.profiler.log_dir = "./fit/profilier/sphere"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Checkpoint # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_path = "./fit/checkpoints/sphere"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 100000  # Always save checkpoint at the end of training

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Model  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.model = ConfigDict()
    cfg.model.name = "AutoDecoder"
    cfg.model.n_hidden = 23
    cfg.model.rff_dim = 64
    cfg.model.center = 0.5

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  UMAP # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.umap = ConfigDict()
    cfg.umap.umap_embeddings_path = "./datasets/sphere/charts/umap_embeddings.npy"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "Sphere"
    cfg.dataset.charts_path = "./datasets/sphere/charts"
    cfg.dataset.distance_matrix_path = "./datasets/sphere/charts/distance_matrix.npy"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Training  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ConfigDict()
    cfg.train.warmup_steps = 20000
    cfg.train.num_steps = 60001
    cfg.train.lr = 1e-4
    cfg.train.reg_lambda = 2.
    cfg.train.weight_decay = 1e-3
    cfg.train.reg_lambda_decay = 0.9999
    cfg.train.reg = "reg+geo"
    cfg.train.noise_scale_riemannian = 0.1
    cfg.train.lambda_geo_loss = 50.

    return cfg
