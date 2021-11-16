from datasets import heateq_2d_square
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import animate_2d_image, heatmap_1d, wireframe_1d
from reduction import truncated_svd_1d, svd_1d
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    n_modes = 3

    u, t, x, M = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=1.49,
        Δt=0.01,
        x_start=0.0,
        x_end=1.0,
        Δx=0.01,
        α=-0.5,
    )
    # TODO establish (Nx x Nt) instead of (Nt x Nx) in plotting functions
    u = u.T
    # end TODO

    U, S, V = np.linalg.svd(u, full_matrices=False)
    z = U.T @ u
    u_reconstructed_full = U @ z
    assert np.allclose(u, u_reconstructed_full)

    Ut = U[:, : n_modes + 1]
    zt = Ut.T @ u
    u_reconstructed_truncated = Ut @ zt

    # ================== Projection to z and simulation ===========
    Mz = Ut.T @ M @ Ut
    Mzi = np.linalg.inv(Mz)
    z0 = Ut.T @ u[:, 0]
    z = [z0]
    cur = z0
    next = cur
    for _ in t:
        next = Mzi @ cur
        z.append(next)
        cur = next

    Z = np.stack(z, axis=1)
    u_projected = Ut @ Z

    # TODO establish (Nx x Nt) instead of (Nt x Nx) in plotting functions
    u = u.T
    u_reconstructed_full = u_reconstructed_full.T
    u_reconstructed_truncated = u_reconstructed_truncated.T
    u_projected = u_projected.T
    # end TODO

    # ================== plotting ======================

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

    fig, ax = heatmap_1d(u_projected, x, t)
    ax.set_title(f"projected and simulated, N = {n_modes}")
    fig, ax = wireframe_1d(u_projected, x, t)
    ax.set_title(f"projected and simulated, N = {n_modes}")
    fig, ax = plt.subplots()
    ax.stem(S)

    plt.show()
