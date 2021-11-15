from numpy.core.numeric import full
from numpy.lib.twodim_base import diag
from datasets import heateq_2d_square
from plotting import animate_2d_image
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    n_components = 2

    u, t, x, y = heateq_2d_square(t_start=0.0, t_end=1.0, Δt=0.01, x_start=0.0,
                                  x_end=1.0, Δx=0.01, y_start=0.0, y_end=1.0, Δy=0.01, α=-0.5)

    # reshape from (Nt x Nx x Ny) to (Nt x Nx*Ny)
    u_flattened = u.reshape(t.shape[0], -1)

    U, Σ, V = svd(u_flattened, full_matrices=False)
    u_reconstructed_full = np.dot(U * Σ, V).reshape(*u.shape)
    u_reconstructed_truncated = np.dot(
        U[:, :n_components] * Σ[:n_components], V[:n_components, :]).reshape(*u.shape)
    assert u_reconstructed_full.shape == u_reconstructed_truncated.shape, "reconstrtion of full SVD and truncated SVD should have identical dimensions"

    fig, ax = plt.subplots()
    ax.stem(Σ)

    fig1, ax1, ani1 = animate_2d_image(u, t)
    ax1.set_title("original u(x,y,t)")

    fig2, ax2, ani2 = animate_2d_image(u_reconstructed_full, t)
    ax2.set_title("reconstructed u(x,y,t)")

    fig3, ax3, ani3 = animate_2d_image(u_reconstructed_truncated, t)
    ax3.set_title(f"reconstructed truncated u(x,y,t), N={n_components}")

    plt.show()
