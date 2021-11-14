import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.core.numeric import zeros_like
from tqdm import tqdm


if __name__ == "__main__":

    t_start = 0.0
    t_end = 1.0
    Δt = 0.01
    t = np.arange(t_start, t_end, Δt)

    x_start = 0.0
    x_end = 1.0
    Δx = 0.01
    y_start = 0.0
    y_end = 1.0
    Δy = 0.01
    x = np.arange(x_start, x_end, Δx)
    y = np.arange(y_start, y_end, Δy)
    α = -0.5

    n_steps = t.shape[0]

    u0 = np.zeros((x.shape[0], y.shape[0]))
    u0[45:55, 45:55] = 1.0

    # ------------------ solving ------------------------

    u = [u0]
    cur = u0

    for _ in tqdm(range(n_steps), desc="stepping"):
        dudx = np.gradient(cur,Δx,axis=0)
        dudxx = np.gradient(dudx, axis=0)
        dudy = np.gradient(cur,Δy,axis=1)
        dudyy = np.gradient(dudy, axis=1)

        dudt = -α*(dudxx+dudyy)*Δt
        next = cur + dudt
        u.append(next)
        cur = next

    u = np.stack(u)

    # =================== plotting ===================

    # ------------------- Initial and terminal ----------------

    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="jet",origin="lower")
    fig.canvas.manager.set_window_title(f'solution t={t_start}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im)


    fig, ax = plt.subplots()
    im = ax.imshow(u[-1], cmap="jet",origin="lower")
    fig.canvas.manager.set_window_title(f'solution t={t_end}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im)

    # ------------------ animated heatmap ---------------------

    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="jet",origin="lower")
    time_text = ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im)

    def update(i):
        im.set_array(u[i])
        time_text.set_text(f"t={t[i]:.2f}")
        return im, time_text

    ani = FuncAnimation(plt.gcf(), update, frames=range(
        n_steps), interval=5, blit=True)

    plt.show()
