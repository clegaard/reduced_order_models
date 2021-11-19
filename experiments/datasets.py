import numpy as np
from tqdm import tqdm
import scipy.linalg


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

        dudt = -α * (dudxx + dudyy) * Δt
        next = cur + dudt
        u.append(next)
        cur = next

    u = np.stack(u)

    return u, t, x, y


def heateq_1d_square_explict_euler(t_start, t_end, Δt, x_start, x_end, Δx, α):
    t = np.arange(t_start, t_end, Δt)
    x = np.arange(x_start, x_end, Δx)

    n_steps = t.shape[0]

    u0 = np.zeros_like(x)
    u0[45:55] = 1.0

    # ------------------ solving ------------------------

    u = [u0]
    cur = u0

    for _ in tqdm(range(n_steps), desc="stepping"):
        dudx = np.gradient(cur, Δx)
        dudxx = np.gradient(dudx)
        dudt = -α * dudxx * Δt
        next = cur + dudt
        u.append(next)
        cur = next

    u = np.vstack(u)

    return u, t, x


def discrete_laplacian_1d(n):
    K = np.zeros((n, n))
    K[0, 0:3] = [1, -2, 1]
    K[-1, -3:] = [1, -2, 1]
    for i in range(1, K.shape[0] - 1):
        K[i, i - 1 : i + 2] = [1, -2, 1]

    return K


def lu_inverse(x):
    pl, u = scipy.linalg.lu(x, permute_l=True)
    pli = np.linalg.inv(pl)
    ui = np.linalg.inv(u)

    return pli, ui


def heateq_1d_square_explict_euler_matrix(t_start, t_end, Δt, x_start, x_end, Δx, α):
    t = np.arange(t_start, t_end, Δt)
    x = np.arange(x_start, x_end, Δx)

    n_steps = t.shape[0]

    u0 = np.zeros_like(x)
    u0[45:55] = 1.0

    # ------------------ solving ------------------------

    u = [u0]
    cur = u0

    # construct discrete laplacian operator to evaluate second derivatives
    K = discrete_laplacian_1d(x.shape[0])
    K = (K * -α * Δt) / Δx
    M = np.identity(K.shape[0]) + K

    for _ in tqdm(range(n_steps), desc="stepping"):
        dudx = np.gradient(cur, Δx)
        dudxx = np.gradient(dudx)
        dudt = -α * dudxx * Δt
        dudxx_k = K @ cur
        # assert np.allclose(dudt, dudxx_k)
        next = M @ cur
        u.append(next)
        cur = next

    u = np.vstack(u)

    return u, t, x


def heateq_1d_square_implicit_euler_matrix(t_start, t_end, Δt, x_start, x_end, Δx, α):
    t = np.arange(t_start, t_end, Δt)
    x = np.arange(x_start, x_end, Δx)

    n_steps = t.shape[0]

    u0 = np.zeros_like(x)
    u0[45:55] = 1.0

    u = [u0]
    cur = u0

    K = discrete_laplacian_1d(x.shape[0])
    γ = α * Δt / Δx ** 2
    M = np.identity(K.shape[0]) - K * γ

    li, ui = lu_inverse(M)

    for _ in tqdm(range(n_steps), desc="stepping"):
        next = ui @ li @ cur
        u.append(next)
        cur = next

    u = np.vstack(u)

    return u, t, x, M
