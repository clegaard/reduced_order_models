import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_2d_image(u, t):
    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="jet", origin="lower")
    time_text = ax.text(
        0.5,
        0.85,
        "",
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        transform=ax.transAxes,
        ha="center",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im)

    def update(i):
        im.set_array(u[i])
        time_text.set_text(f"t={t[i]:.2f}")
        return im, time_text

    ani = FuncAnimation(
        plt.gcf(), update, frames=range(t.shape[0]), interval=5, blit=True
    )

    return fig, ax, ani


def waterfall_1d(u, x, t):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(t.shape[0]):
        Y = np.ones_like(x) * t[i]
        Z = u[i, :]
        ax.plot(x, Y, Z)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t)")

    return fig, ax


def _wireframe_and_surface_1d(u, x, t, type):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    X = []
    Y = []
    Z = []

    for i in range(t.shape[0]):
        X.append(x)
        Y.append(np.ones_like(x) * t[i])
        Z.append(u[i, :])

    X = np.stack(X)
    Y = np.stack(Y)
    Z = np.stack(Z)

    if type == "wireframe":
        ax.plot_wireframe(X, Y, Z)
    elif type == "surface":
        ax.plot_surface(X, Y, Z)
    else:
        raise ValueError()

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t)")

    return fig, ax


def wireframe_1d(u, x, t):
    return _wireframe_and_surface_1d(u, x, t, type="wireframe")


def surface_1d(u, x, t):
    return _wireframe_and_surface_1d(
        u,
        x,
        t,
        type="surface",
    )


def heatmap_1d(u, x, t):

    fig, ax = plt.subplots()
    im = ax.imshow(u, cmap="jet", origin="lower")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(im)

    return fig, ax
