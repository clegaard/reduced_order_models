import numpy as np
from tqdm import tqdm


def heateq_2d_square(t_start, t_end, Δt, x_start, x_end, Δx, y_start, y_end, Δy, α):
    t = np.arange(t_start, t_end, Δt)

    x = np.arange(x_start, x_end, Δx)
    y = np.arange(y_start, y_end, Δy)

    n_steps = t.shape[0]

    u0 = np.zeros((x.shape[0], y.shape[0]))
    u0[45:55, 45:55] = 1.0

    # ------------------ solving ------------------------

    u = [u0]
    cur = u0

    for _ in tqdm(range(n_steps), desc="stepping"):
        dudx = np.gradient(cur, Δx, axis=0)
        dudxx = np.gradient(dudx, axis=0)
        dudy = np.gradient(cur, Δy, axis=1)
        dudyy = np.gradient(dudy, axis=1)

        dudt = -α*(dudxx+dudyy)*Δt
        next = cur + dudt
        u.append(next)
        cur = next

    u = np.stack(u)

    return u, t, x, y
