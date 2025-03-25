import numpy as np
from torch.utils import data
from stl import mesh
from scipy.spatial import Delaunay
import open3d as o3d
import numpy as np


class StanfordBunny(data.Dataset):
    def __init__(
        self, path, scale, points_per_unit_area=None, subset_cardinality=None, seed=42
    ):
        """
        Args:
            path (str): path to the .npy file containing the .npy file
            scale (float): scale factor to apply to the data
            points_per_unit_area (float): Number of points to sample per unit area
            subset_cardinality (int): Number of points to sample from the final dataset
            seed (int): Random seed

        """

        if path.endswith(".stl"):
            bunny_mesh = mesh.Mesh.from_file(path)
            points = sample_points_from_mesh(bunny_mesh, points_per_unit_area)
        else:
            points = np.load(path)

        self.data = points * scale
        if subset_cardinality is not None:
            rng = np.random.default_rng(seed)
            if subset_cardinality > len(self.data):
                indices = rng.choice(len(self.data), size=len(self.data), replace=False)
            else:
                indices = rng.choice(
                    len(self.data), size=subset_cardinality, replace=False
                )
            self.data = self.data[indices]

        # self.data = self.data[(self.data[:, 1] > 0.4)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_stl_to_points(stl_path, points_per_unit_area=40):
    """
    Load STL file and convert to point cloud with points sampled based on triangle area.

    Args:
        stl_path (str): Path to the STL file
        points_per_unit_area (float): Number of points to sample per unit area

    Returns:
        numpy.ndarray: Point cloud array with shape (N, 3)
    """
    # Load the STL file
    mesh_data = mesh.Mesh.from_file(stl_path)

    # Get vertices of all triangles
    vertices = mesh_data.vectors.reshape(-1, 3)

    # Initialize list to store all points
    all_points = [vertices]

    # Sample additional points on each triangle
    for triangle in mesh_data.vectors:
        # Get triangle vertices
        v1, v2, v3 = triangle

        # Calculate triangle area using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2

        # Calculate number of points to sample based on area
        num_samples = max(1, int(area * points_per_unit_area))

        # Generate random barycentric coordinates
        r1 = np.random.random((num_samples, 1))
        r2 = np.random.random((num_samples, 1))

        # Ensure the random points lie within the triangle
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        # Calculate barycentric coordinates
        a = 1 - r1 - r2
        b = r1
        c = r2

        # Generate points using barycentric coordinates
        points = a * v1 + b * v2 + c * v3
        all_points.append(points)

    # Combine all points into single array
    point_cloud = np.vstack(all_points)

    return point_cloud


def point_cloud_to_mesh(points):
    """
    Convert a 3D point cloud to a mesh using Ball Pivoting Algorithm.

    Args:
        points (numpy.ndarray): Point cloud array with shape (N, 3)

    Returns:
        o3d.geometry.TriangleMesh: Mesh object created from the point cloud
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals()

    # Create a mesh using Ball Pivoting Algorithm
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )

    return bpa_mesh


def sample_points_from_mesh(bunny_mesh, points_per_unit_area=2):
    """
    Sample points from the mesh triangles.

    Args:
        bunny_mesh (mesh.Mesh): Mesh object
        points_per_unit_area (float): Number of points to sample per unit area

    Returns:
        numpy.ndarray: Point cloud array with shape (N, 3)
    """
    all_points = []

    for triangle in bunny_mesh.vectors:
        # Get triangle vertices
        v1, v2, v3 = triangle

        # Calculate triangle area using cross product
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal) / 2

        # Calculate number of points to sample based on area
        num_samples = max(1, int(area * points_per_unit_area))

        # Generate random barycentric coordinates
        r1 = np.random.random((num_samples, 1))
        r2 = np.random.random((num_samples, 1))

        # Ensure the random points lie within the triangle
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        # Calculate barycentric coordinates
        a = 1 - r1 - r2
        b = r1
        c = r2

        # Generate points using barycentric coordinates
        points = a * v1 + b * v2 + c * v3
        all_points.append(points)

    # Combine all points into single array
    point_cloud = np.vstack(all_points)

    return point_cloud
