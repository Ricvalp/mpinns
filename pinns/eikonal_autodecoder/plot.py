import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def plot_domains(x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, name=None):
    num_plots = len(x)
    cols = 4  # You can adjust the number of columns based on your preference
    rows = (num_plots + cols - 1) // cols  # Calculate required rows

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    # Ensure ax is a 2D array for easy indexing
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    for i in range(num_plots):
        # Calculate row and column index for the plot
        row, col = divmod(i, cols)
        ax[row][col].set_title(f"Chart {i}")
        scatter = ax[row][col].scatter(x[i], y[i], s=3, c='b')
        scatter_bcs = ax[row][col].scatter(bcs_x[i], bcs_y[i], s=50, c=bcs[i], label="BCs")
        # Add colorbar for boundary conditions
        if len(np.unique(bcs[i])) > 1:  # Only add colorbar if there are multiple colors
            fig.colorbar(scatter_bcs, ax=ax[row][col], orientation="vertical", label="BC Value")

        # Plot boundaries for current chart
        if i in boundaries_x:
            for other_chart, boundary_x in boundaries_x[i].items():
                ax[row][col].scatter(
                    boundary_x,
                    boundaries_y[i][other_chart],
                    s=10,
                    label=f"boundary {i}-{other_chart}",
                )

        ax[row][col].legend(loc="best")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_with_metric(x, y, sqrt_det_g, d_params, name=None):
    num_plots = len(x)
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_plots == 1:
        ax = [[ax]]
    elif cols == 1 or rows == 1:
        ax = ax.reshape(-1, cols)

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(num_plots):
        row, col = divmod(i, cols)

        ax[row][col].set_title(f"Chart {i}")
        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))
        scatter = ax[row][col].scatter(x[i], y[i], s=3, c=color_values, cmap="viridis")
        fig.colorbar(scatter, ax=ax[row][col], orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d(x, y, bcs_x, bcs_y, bcs, decoder, d_params, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot")
    
    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    
    # Create a single colorbar for all BC points
    all_bc_values = np.concatenate([bcs[i] for i in range(len(bcs))])
    vmin, vmax = np.min(all_bc_values), np.max(all_bc_values)
    
    for i in range(len(x)):
        p = decoder.apply({"params": decoder_params[i]}, np.stack([x[i], y[i]], axis=1))
        p_bcs = decoder.apply(
            {"params": decoder_params[i]}, np.stack([bcs_x[i], bcs_y[i]], axis=1)
        )
        
        # Plot the domain points
        ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=3,
            alpha=0.5,
            label=f"Chart {i}"
        )
        
        # Plot the boundary conditions with colors
        scatter_bcs = ax.scatter(
            p_bcs[:, 0],
            p_bcs[:, 1],
            p_bcs[:, 2],
            c=bcs[i],
            s=50,
            vmin=vmin,
            vmax=vmax,
            label=f"BCs {i}"
        )
    
    # Add a single colorbar for all boundary conditions
    cbar = fig.colorbar(scatter_bcs, ax=ax, orientation="vertical", label="BC Value")
    
    # Set consistent axes limits
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_domains_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    num_plots = len(x)
    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Chart {i}")
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3)

        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_combined_3d_with_metric(x, y, decoder, sqrt_det_g, d_params, name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Combined 3D Plot with Metric Coloring")

    decoder_params = [jax.tree_map(lambda x: x[i], d_params) for i in range(len(x))]
    for i in range(len(x)):
        # Decode the 2D points to 3D using the decoder function
        points_3d = decoder.apply(
            {"params": decoder_params[i]}, jnp.stack([x[i], y[i]], axis=1)
        )
        x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Calculate the color values using sqrt_det_gs
        color_values = sqrt_det_g(decoder_params[i], jnp.stack([x[i], y[i]], axis=1))

        scatter = ax.scatter(
            x_3d, y_3d, z_3d, c=color_values, cmap="viridis", s=3, label=f"Chart {i}"
        )

    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, orientation="vertical")

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_level_curves(x, y, u_preds, config):
    """
    Plot level curves (contours) for the solution of the Eikonal equation.
    """
    for i in range(len(u_preds)):
        # Flatten the point cloud and scalar values
        points = np.stack([x[i], y[i]], axis=1)
        values = u_preds[i]

        # Create a grid for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(points[:, 0].min(), points[:, 0].max(), 500),
            np.linspace(points[:, 1].min(), points[:, 1].max(), 500),
        )

        # Interpolate the scalar values onto the grid
        grid_u = griddata(points, values, (grid_x, grid_y), method="linear")

        # Plot the level curves
        plt.figure(figsize=(8, 6))
        contour = plt.contour(grid_x, grid_y, grid_u, levels=20, cmap="jet")
        plt.colorbar(contour, label="u value")
        plt.title(f"Level Curves for Chart {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(config.figure_path + f"/level_curves_chart_{i}.png")
        plt.close()


def plot_3d_level_curves(x, y, z, u_preds, config):
    """
    Plot level curves (contours) on a 3D surface for the solution of the Eikonal equation.
    """
    for i in range(len(u_preds)):
        # Prepare the data for the current chart
        points = np.stack([x[i], y[i], z[i]], axis=1)
        values = u_preds[i]

        # Create a grid for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(points[:, 0].min(), points[:, 0].max(), 100),
            np.linspace(points[:, 1].min(), points[:, 1].max(), 100),
        )

        # Interpolate the scalar values onto the grid
        grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method="linear")
        grid_u = griddata(points[:, :2], values, (grid_x, grid_y), method="linear")

        # Plot the 3D surface with level curves
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Level Curves for Chart {i}")

        # Plot the surface
        surf = ax.plot_surface(
            grid_x, grid_y, grid_z, facecolors=plt.cm.jet(grid_u / grid_u.max()), rstride=1, cstride=1, alpha=0.8
        )

        # Add level curves
        contour = ax.contour(
            grid_x, grid_y, grid_u, levels=20, cmap="jet", linestyles="solid", offset=grid_z.min()
        )

        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap="jet")
        mappable.set_array(grid_u)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="u value")

        # Set labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Save the plot
        plt.tight_layout()
        plt.savefig(config.figure_path + f"/3d_level_curves_chart_{i}.png")
        plt.close()