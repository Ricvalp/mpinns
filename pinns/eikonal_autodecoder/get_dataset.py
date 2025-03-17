import numpy as np
import jax

from chart_autoencoder.get_charts import load_charts




def compute_bcs(x, y):

    Y_eg = igl.exact_geodesic(m.verts, m.connectivity, onp.array([0]), onp.arange(m.verts.shape[0]))
    n_nodes = m.verts.shape[0]
    N = 50 # number of data points
    rng_key, subkey = random.split(rng_key)
    idx_train = random.choice(random.PRNGKey(30), n_nodes, (N,), replace = False)
    
    return 


def get_dataset(charts_path, sigma=0.1):

    loaded_charts3d, loaded_boundaries, loaded_boundary_indices, loaded_charts2d = load_charts(
        charts_path, from_autodecoder=True
    )

    x = {}
    y = {}

    for chart_key in loaded_charts2d.keys():
        x[chart_key] = loaded_charts2d[chart_key][:, 0]
        y[chart_key] = loaded_charts2d[chart_key][:, 1]

    boundaries_x = {}
    boundaries_y = {}

    for key in loaded_boundary_indices.keys():
        start_boundary_indices = np.array(loaded_boundary_indices[key])

        starting_chart = key[0]
        starting_chart_points = loaded_charts2d[starting_chart][start_boundary_indices]
        ending_chart = key[1]

        if starting_chart not in boundaries_x:
            boundaries_x[starting_chart] = {}
        if starting_chart not in boundaries_y:
            boundaries_y[starting_chart] = {}

        boundaries_x[starting_chart][ending_chart] = starting_chart_points[:, 0]
        boundaries_y[starting_chart][ending_chart] = starting_chart_points[:, 1]

    bcs = {}

    return x, y, bcs, boundaries_x, boundaries_y, loaded_charts3d
