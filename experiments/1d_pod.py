from datasets import heateq_2d_square
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import animate_2d_image, heatmap_1d, wireframe_1d
from reduction import truncated_svd_1d, svd_1d
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    n_modes = 2

    u, t, x, M = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=0.5,
        Δt=0.01,
        x_start=0.0,
        x_end=1.0,
        Δx=0.01,
        α=-0.5,
    )

    U, S, V = np.linalg.svd(u, full_matrices=False)
    z = U.T @ u
    u_reconstructed_full = U @ z

    assert np.allclose(u, u_reconstructed_full)
    Ut = U[:, :n_modes]

    zt = Ut.T @ u
    u_reconstructed_truncated = Ut @ zt

    # U.T @ M @ U

    fig, ax = heatmap_1d(u, x, t)
    ax.set_title("original")
    fig, ax = wireframe_1d(u, x, t)
    ax.set_title("original")

    fig, ax = heatmap_1d(u_reconstructed_full, x, t)
    ax.set_title("reconstructed full")
    fig, ax = wireframe_1d(u_reconstructed_full, x, t)
    ax.set_title("reconstructed full")

    fig, ax = heatmap_1d(u_reconstructed_truncated, x, t)
    ax.set_title(f"reconstructed truncated, N = {n_modes}")
    fig, ax = wireframe_1d(u_reconstructed_truncated, x, t)
    ax.set_title(f"reconstructed truncated, N = {n_modes}")
    fig, ax = plt.subplots()
    ax.stem(S)

    plt.show()
