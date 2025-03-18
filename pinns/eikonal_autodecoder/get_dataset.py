import numpy as np
from jax import random
import igl

from datasets.utils import Mesh
from chart_autoencoder.get_charts import load_charts


def get_dataset(charts_path, mesh_path, scale, N=100):

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

    bcs_x, bcs_y, bcs = get_eikonal_bcs(
        mesh_path=mesh_path, scale=scale, x=x, y=y, charts3d=loaded_charts3d, N=N
    )

    return x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, loaded_charts3d



def get_eikonal_bcs(mesh_path, scale, x, y, charts3d, N=50, seed=37):
    
    m = Mesh(mesh_path)
    verts, connectivity = m.verts*scale, m.connectivity
        
    Y_eg = igl.exact_geodesic(verts, connectivity, np.array([0]), np.arange(verts.shape[0]))
    n_nodes = verts.shape[0]
    key = random.PRNGKey(seed)
    idx_train = random.choice(key, n_nodes, (N,), replace = False)
    
    Y = Y_eg[idx_train]
    
    bcs_points = verts[idx_train]
    
    bcs_x = {}
    bcs_y = {}
    bcs = {}
    
    for key in charts3d.keys():
        in_indices = np.where(np.isin(bcs_points, charts3d[key]).all(axis=1))
        if len(in_indices) > 0:
            bcs_x[key] = x[key][in_indices]
            bcs_y[key] = y[key][in_indices]
            bcs[key] = Y[in_indices]    

    return bcs_x, bcs_y, bcs