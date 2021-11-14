from numpy.lib.twodim_base import diag
from datasets import heateq_2d_square
from plotting import animate_2d_image
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    u, t, x, y = heateq_2d_square(t_start=0.0, t_end=1.0, Δt=0.01, x_start=0.0,
                                  x_end=1.0, Δx=0.01, y_start=0.0, y_end=1.0, Δy=0.01, α=-0.5)

    # U, Σ, V = svd(u, full_matrices=True)

    # ax.stem(Σ)
    # ax.set_title("Singular values summed along time")
    # u_reconstructed = U @ Σ @ V.T

    # fig, ax = plt.subplots()

    fig1, ax1, ani1 = animate_2d_image(u, t)
    # fig2, ax2, ani2 = animate_2d_image(u_reconstructed, t)

    plt.show()
