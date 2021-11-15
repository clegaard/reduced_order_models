import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import zeros_like
from tqdm import tqdm

from datasets import (
    heateq_1d_square_explict_euler_matrix,
)


if __name__ == "__main__":

    t_start = 0.0
    t_end = 1.0
    Δt = 0.01

    x_start = 0.0
    x_end = 1.0
    Δx = 0.01
    α = -0.1

    # ------------------ solving ------------------------
    u, t, x = heateq_1d_square_explict_euler_matrix(
        t_start=t_start, t_end=t_end, Δt=Δt, x_start=x_start, x_end=x_end, Δx=Δx, α=α
    )

    n_steps = t.shape[0]

    # =================== plotting ===================

    # ------------------ waterfall --------------------

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in tqdm(range(n_steps), desc="plotting"):
        Y = np.ones_like(x) * t[i]
        Z = u[i, :]
        ax.plot(x, Y, Z)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t)")

    # ------------------ heatmap ---------------------

    fig, ax = plt.subplots()
    im = ax.imshow(u, cmap="jet", origin="lower")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(im)

    plt.show()
