import numpy as np


def svd_1d(u):
    return truncated_svd_1d(u, -1)


def truncated_svd_1d(u, n_modes):
    U, Σ, V = np.linalg.svd(u, full_matrices=False)

    Ut = U[:, :n_modes]
    Σt = Σ[:n_modes]
    Vt = V[:n_modes, :]

    u_reconstructed_truncated = np.dot(Ut * Σt, Vt).reshape(*u.shape)

    return u_reconstructed_truncated, Ut, Σt, Vt


def galerkin_projection_1d(
    M,
):
    pass
