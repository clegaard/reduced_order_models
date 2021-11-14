import matplotlib.pyplot as plt
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
    x = np.arange(x_start, x_end, Δx)
    α = -0.5

    n_steps = t.shape[0]

    u0 = zeros_like(x)
    u0[45:55] = 1.0

    # ------------------ solving ------------------------

    u = [u0]
    cur = u0

    for _ in tqdm(range(n_steps),desc="stepping"):
        dudx = np.gradient(cur, Δx)
        dudxx = np.gradient(dudx)
        dudt = -α*dudxx*Δt
        next = cur + dudt
        u.append(next)
        cur = next

    u = np.vstack(u)

    # =================== plotting ===================
    
    # ------------------ waterfall --------------------
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in tqdm(range(n_steps),desc="plotting"):
        Y = np.ones_like(x)*t[i]
        Z = u[i, :]
        ax.plot(x, Y, Z)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t)")

    # ------------------ heatmap ---------------------

    fig, ax = plt.subplots()
    im = ax.imshow(u,cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.colorbar(im)

    plt.show()
