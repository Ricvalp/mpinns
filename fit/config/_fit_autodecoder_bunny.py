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
    cfg.wandb.use = False
    cfg.wandb.project = "chart-autoencoder"
    cfg.wandb.name = "bunny"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_charts_every = 20000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Profiler # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.profiler = ConfigDict()
    cfg.profiler.start_step = 300
    cfg.profiler.end_step = 305
    cfg.profiler.log_dir = "./fit/profilier/bunny"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Checkpoint # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_path = "./fit/checkpoints/bunny"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 100000

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # #  Model  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.model = ConfigDict()
    cfg.model.name = "AutoDecoder"
    cfg.model.n_hidden = 32
    cfg.model.rff_dim = 128
    cfg.model.center = 0.5

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  UMAP # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.umap = ConfigDict()
    cfg.umap.umap_scale = 0.1
    cfg.umap.use_existing_umap_embeddings = False
    cfg.umap.umap_embeddings_path = "./datasets/bunny/charts/umap_embeddings.npy"
    cfg.umap.n_neighbors = 5
    cfg.umap.learning_rate = 1.0
    cfg.umap.min_dist = 0.1
    cfg.umap.random_state = 42
    cfg.umap.n_components = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "StanfordBunny"
    cfg.dataset.path = (
        "./datasets/stl_files/Stanford_Bunny.stl"  # "/stanford_bunny.npy"
    )
    cfg.dataset.scale = 0.05
    cfg.dataset.seed = 37
    cfg.dataset.points_per_unit_area = 2
    cfg.dataset.subset_cardinality = 300000
    cfg.dataset.load_existing_charts = True
    cfg.dataset.charts_path = "./datasets/bunny/charts"
    cfg.dataset.use_existing_distance_matrix = False
    cfg.dataset.distance_matrix_path = "./datasets/bunny/charts/distance_matrix.npy"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"
    cfg.charts.center = 0.5
    cfg.charts.min_dist = 1.0
    cfg.charts.nearest_neighbors = 10
    cfg.charts.charts_to_train = [18, 19, 22]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # Charts to refine  # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts_to_refine = ConfigDict()
    cfg.charts_to_refine.chart_to_refine = 45
    cfg.charts_to_refine.min_dist = 0.2
    cfg.charts_to_refine.nearest_neighbors = 10
    cfg.charts_to_refine.save_charts = True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Training  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.train = ConfigDict()
    cfg.train.warmup_steps = 20000
    cfg.train.num_steps = 20001
    cfg.train.batch_size = 64
    cfg.train.lr = 1e-4
    cfg.train.reg_lambda = 1.0
    cfg.train.weight_decay = 1e-3
    cfg.train.reg_lambda_decay = 0.9999
    cfg.train.reg = "reg+geo"
    cfg.train.noise_scale_riemannian = 0.1
    return cfg
