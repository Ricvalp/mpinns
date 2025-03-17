import numpy as np
import jax

from chart_autoencoder.get_charts import load_charts


def initial_conditions_spike(x, y, x0=0.5, y0=0.5, sigma=0.4):
    return 10 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def initial_conditions_zero(x, y):
    return 0.0 * x


def get_dataset(encoder, e_params, charts_path, sigma=0.1):

    charts3d, boundaries3d = load_charts(charts_path)

    x = {}
    y = {}

    encoder_params = [
        jax.tree_map(lambda x: x[i], e_params) for i in range(len(charts3d))
    ]

    for i, (chart_key, encoder_param) in enumerate(
        zip(charts3d.keys(), encoder_params)
    ):
        z = encoder.apply({"params": encoder_param}, charts3d[chart_key])
        x[chart_key] = jax.device_get(z[:, 0])
        y[chart_key] = jax.device_get(z[:, 1])

    boundaries_x = {}
    boundaries_y = {}
    for key in boundaries3d.keys():
        boundary = boundaries3d[key]

        starting_chart = key[0]
        starting_chart_points = encoder.apply(
            {"params": encoder_params[starting_chart]}, boundary
        )
        ending_chart = key[1]
        ending_chart_points = encoder.apply(
            {"params": encoder_params[ending_chart]}, boundary
        )
        if starting_chart not in boundaries_x:
            boundaries_x[starting_chart] = {}
        if starting_chart not in boundaries_y:
            boundaries_y[starting_chart] = {}

        boundaries_x[starting_chart][ending_chart] = jax.device_get(
            starting_chart_points[:, 0]
        )
        boundaries_y[starting_chart][ending_chart] = jax.device_get(
            starting_chart_points[:, 1]
        )

        if ending_chart not in boundaries_x:
            boundaries_x[ending_chart] = {}
        if ending_chart not in boundaries_y:
            boundaries_y[ending_chart] = {}

        boundaries_x[ending_chart][starting_chart] = jax.device_get(
            ending_chart_points[:, 0]
        )
        boundaries_y[ending_chart][starting_chart] = jax.device_get(
            ending_chart_points[:, 1]
        )

    ics = {}
    ics[0] = initial_conditions_spike(x[0], y[0], sigma=sigma)
    for i in range(1, len(x)):
        ics[i] = initial_conditions_zero(x[i], y[i])

    return x, y, ics, boundaries_x, boundaries_y
