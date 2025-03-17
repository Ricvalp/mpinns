from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Wandb  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "chart-autoencoder-rff"
    cfg.wandb.name = "propeller-rff"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_charts_every = 10000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Profiler # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.profiler = ConfigDict()
    cfg.profiler.start_step = 300
    cfg.profiler.end_step = 305
    cfg.profiler.log_dir = "./fit/profilier/propeller"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Plot # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.plot = ConfigDict()
    cfg.plot.plot_boundaries = False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Checkpoint # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_path = "./fit/checkpoints/propeller"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 100000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Model  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.model = ConfigDict()
    cfg.model.name = "AutoDecoder"
    cfg.model.n_hidden = 8
    cfg.model.rff_dim = 8
    cfg.model.center = 0.5

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  UMAP  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.umap = ConfigDict()
    cfg.umap.scale = 0.01
    cfg.umap.umap_scale = 0.01
    cfg.umap.use_existing_umap_embeddings = True
    cfg.umap.umap_embeddings_path = "./datasets/propeller/charts/umap_embeddings.npy"
    cfg.umap.n_neighbors = 5
    cfg.umap.learning_rate = 0.1
    cfg.umap.min_dist = 0.0
    cfg.umap.random_state = 42
    cfg.umap.n_components = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "Propeller"
    cfg.dataset.path = "./datasets/stl_files/titanic_propeller_1.stl"
    cfg.dataset.scale = 0.02
    cfg.dataset.seed = 37
    cfg.dataset.points_per_unit_area = 2
    cfg.dataset.subset_cardinality = None
    cfg.dataset.load_existing_charts = True
    cfg.dataset.save_charts = True
    cfg.dataset.use_existing_distances_matrix = False
    cfg.dataset.charts_path = "./datasets/propeller/charts"
    cfg.dataset.distances_matrices_path = (
        "./datasets/propeller/charts/distances_matrices.npy"
    )
    cfg.dataset.save_distances_matrix = True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"  # "louvain_clustering"
    cfg.charts.center = 0.5
    cfg.charts.min_dist = 0.9
    cfg.charts.nearest_neighbors = 10

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Training  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ConfigDict()
    cfg.train.warmup_steps = 10000
    cfg.train.num_steps = 50000
    cfg.train.batch_size = 64
    cfg.train.lr = 1e-4
    cfg.train.reg_lambda = 1e-3
    cfg.train.weight_decay = 1e-3
    cfg.train.reg = "none"

    return cfg
