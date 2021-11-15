import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import zeros_like
from tqdm import tqdm

from datasets import (
    heateq_1d_square_explict_euler_matrix,
    heateq_1d_square_implicit_euler_matrix,
)

from plotting import waterfall_1d, heatmap_1d, wireframe_1d, surface_1d


if __name__ == "__main__":

    t_start = 0.0
    t_end = 1.0
    Δt = 0.01

    x_start = 0.0
    x_end = 1.0
    Δx = 0.01
    α = -0.1

    # ------------------ solving ------------------------

    u, t, x = heateq_1d_square_implicit_euler_matrix(
        t_start=t_start, t_end=t_end, Δt=Δt, x_start=x_start, x_end=x_end, Δx=Δx, α=α
    )
    n_steps = t.shape[0]

    # =================== plotting ===================

    # fig, ax = waterfall_1d(u, x, t)

    fig, ax = surface_1d(u, x, t)

    fig, ax = heatmap_1d(u, x, t)

    plt.show()
