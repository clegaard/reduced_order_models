import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_2d_image(u, t):
    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="jet", origin="lower")
    time_text = ax.text(0.5, 0.85, "", bbox={
                        'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im)

    def update(i):
        im.set_array(u[i])
        time_text.set_text(f"t={t[i]:.2f}")
        return im, time_text

    ani = FuncAnimation(plt.gcf(), update, frames=range(
        t.shape[0]), interval=5, blit=True)

    return fig, ax, ani
