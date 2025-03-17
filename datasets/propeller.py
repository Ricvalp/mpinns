import numpy as np
from stl import mesh
from torch.utils import data


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


def plot_point_cloud(points, save_html=None, save_png=None):
    """
    Plot point cloud using both Plotly (HTML) and matplotlib (PNG).

    Args:
        points (numpy.ndarray): Point cloud array with shape (N, 3)
        save_html (str, optional): Path to save interactive HTML plot
        save_png (str, optional): Path to save static PNG plot
    """
    # Plotly plot (interactive HTML)
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=points[:, 2],  # color by z-coordinate
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),  # preserve aspect ratio
        title="3D Point Cloud Visualization",
    )

    if save_html:
        fig.write_html(save_html)

    # Matplotlib plot (static PNG)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", s=1
    )

    plt.colorbar(scatter)
    ax.set_title("3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close()


class Propeller(data.Dataset):
    def __init__(self, path, scale, points_per_unit_area, subset_cardinality, seed=42):
        """
        Args:
            path (str): path to the .npy file containing the .npy file
            scale (float): scale factor
            points_per_unit_area (float): number of points per unit area
            subset_cardinality (int): number of points to sample
            seed (int): random seed
        """

        self.data = (
            load_stl_to_points(path, points_per_unit_area=points_per_unit_area) * scale
        )
        if subset_cardinality is not None:
            rng = np.random.default_rng(seed)
            if subset_cardinality > len(self.data):
                indices = rng.choice(len(self.data), size=len(self.data), replace=False)
            else:
                indices = rng.choice(
                    len(self.data), size=subset_cardinality, replace=False
                )
            self.data = self.data[indices]

        # Center the data at (0, 0, 0)
        center = np.mean(self.data, axis=0)
        self.data = self.data - center

        self.connectivity = None

    def save_dataset(self, dataset_dir):
        np.save(dataset_dir / "propeller.npy", self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Example usage
if __name__ == "__main__":
    # Replace with your STL file path
    stl_file = "datasets/stl_files/propellerimprovedfins5fins.stl"

    # Generate point cloud with 10 additional points per triangle
    points = load_stl_to_points(stl_file, points_per_unit_area=1)

    print(f"Generated point cloud with {len(points)} points")

    # Save points to file
    np.save("point_cloud.npy", points)

    # Plot and save visualizations
    plot_point_cloud(
        points,
        save_html="point_cloud_visualization.html",
        save_png="point_cloud_visualization.png",
    )
