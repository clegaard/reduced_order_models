import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_
import scipy
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau


def discrete_laplacian_1d(n, dtype):
    K = torch.zeros((n, n), dtype=dtype)
    diag = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)
    K[0, 0:3] = diag
    K[-1, -3:] = diag
    for i in range(1, K.shape[0] - 1):
        K[i, i - 1 : i + 2] = diag

    return K


def lu_inverse(x):
    pl, u = scipy.linalg.lu(x, permute_l=True)
    pli = torch.linalg.inv(torch.tensor(pl))
    ui = torch.linalg.inv(torch.tensor(u))

    return pli, ui


def batch_mult(K, X):
    return torch.matmul(X[:, None, :, :].T, K.T).T.squeeze(1)


def train_model(x, γ, n_train_iterations, mode, regularize_diagonal, regularize_l1):
    K = torch.empty(
        (x.shape[0], x.shape[0]),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    xavier_uniform_(K)
    opt = torch.optim.Adam([K], lr=lr)

    scheduler = ReduceLROnPlateau(opt, "min")
    losses = []

    def K_func():
        if regularize_diagonal:
            return -K.T @ K
        else:
            return K

    for _ in tqdm(range(n_train_iterations), desc="training explicit"):

        if mode == "explicit":
            x_next = x + batch_mult(K_func() * γ, x)
            loss = F.mse_loss(x_next[:, :-1], x[:, 1:])
        elif mode == "implicit":
            x_prev = x - batch_mult(K_func() * γ, x)
            loss = F.mse_loss(x_prev[:, 1:], x[:, :-1])
        else:
            raise ValueError("invalid value for mode")

        if regularize_l1:
            loss = loss + 0.000001 * torch.norm(K, p=1)

        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()
        scheduler.step(loss)

    return K_func(), losses


def simulate(M, t, x0):
    next = None
    x = [x0]
    cur = x0
    for _ in range(len(t) - 1):
        next = batch_mult(M, cur)
        x.append(next)
        cur = next
    x = torch.concat(x, dim=1)
    return x


if __name__ == "__main__":

    n_train_iterations = 30000
    n_experiments_train = 1
    n_experiments_validate = 0
    n_modes = 10
    device = "cuda"
    dtype = torch.float32

    Δt = 0.01
    α = 0.005
    Δx = 0.01
    lr = 1e-3
    t_start = 0.0
    t_end = 2.0
    x_start = 0.0
    x_end = 1.0
    regularize_diagonal = True

    # u, t, x, K = heateq_1d_square_implicit_euler_matrix(
    #     t_start=0.0,
    #     t_end=2.0,
    #     Δt=Δt,
    #     x_start=0.0,
    #     x_end=1.0,
    #     Δx=Δx,
    #     α=α,
    # )

    # ===================== Generate Data ==============================

    t = torch.arange(t_start, t_end, Δt, dtype=dtype)
    x = torch.arange(x_start, x_end, Δx, dtype=dtype)

    n_steps = t.shape[0]

    u0 = torch.rand(
        (*x.shape, n_experiments_train + n_experiments_validate), dtype=dtype
    )
    u = [u0]
    cur = u0

    K = discrete_laplacian_1d(x.shape[0], dtype)
    γ = α * Δt / Δx ** 2
    M = torch.eye(K.shape[0]) - K * γ

    li, ui = lu_inverse(M)

    for _ in tqdm(range(n_steps - 1), desc="stepping"):
        next = ui @ li @ cur
        u.append(next)
        cur = next

    u = torch.stack(u, dim=1).to(device)  # Nx x Nt x Ne
    u_flat = rearrange(u, "nx nt ne -> nx (nt ne)")

    # ================== Dimensionality Reduction ======================
    U, S, V = torch.linalg.svd(u_flat, full_matrices=False)

    z = batch_mult(U.T, u)
    u_reconstructed_full = batch_mult(U, z)
    Ut = U[:, :n_modes]
    zt = batch_mult(Ut.T, u)

    u_reconstructed_truncated = batch_mult(Ut, zt)
    K_reduced = Ut.T.cpu() @ K @ Ut.cpu()

    # ========================== Training =============================

    # =========== full system -> forward euler ===========

    K_full_explicit, losses_full_explicit = train_model(
        u, γ, n_train_iterations, "explicit", True, regularize_l1=False
    )

    M_full_explicit = (
        torch.eye(K_full_explicit.shape[0], device=device) + K_full_explicit * γ
    )

    x_full_explicit = simulate(M_full_explicit, t, x0=u[:, :1, :])

    x_full_explicit = x_full_explicit.detach().cpu()
    K_full_explicit = K_full_explicit.detach().cpu()

    # =========== full system -> backward euler ===========

    K_full_implicit, losses_full_implicit = train_model(
        u,
        γ,
        n_train_iterations,
        "implicit",
        regularize_diagonal=regularize_diagonal,
        regularize_l1=False,
    )

    M_full_implicit = torch.linalg.inv(
        torch.eye(K_full_implicit.shape[0], device=device) - K_full_implicit * γ
    )

    x_full_implicit = simulate(M_full_implicit, t, u[:, :1])

    K_full_implicit = K_full_implicit.detach().cpu()
    x_full_implicit = x_full_implicit.detach().cpu()

    # =========== reduced system -> forward euler ===========

    K_reduced_explicit, losses_reduced_explicit = train_model(
        zt,
        γ,
        n_train_iterations,
        "explicit",
        regularize_diagonal=regularize_diagonal,
        regularize_l1=False,
    )

    M_reduced_explicit = (
        torch.eye(K_reduced_explicit.shape[0], device=device) + K_reduced_explicit * γ
    )

    zs = simulate(M_reduced_explicit, t, zt[:, :1])
    x_reduced_explicit = batch_mult(Ut, zs)

    x_reduced_explicit = x_reduced_explicit.detach().cpu()
    K_reduced_explicit = K_reduced_explicit.detach().cpu()

    # ============= Reduced system -> Backward euler =================

    K_reduced_implicit, losses_reduced_implicit = train_model(
        zt,
        γ,
        n_train_iterations,
        "implicit",
        regularize_diagonal=False,
        regularize_l1=False,
    )

    M_reduced_implicit_inv = torch.linalg.inv(
        torch.eye(K_reduced_implicit.shape[0], device=device) - K_reduced_implicit * γ
    )

    zs = simulate(M_reduced_implicit_inv, t, zt[:, :1])
    x_reduced_implicit = batch_mult(Ut, zs)

    K_reduced_implicit = K_reduced_implicit.detach().cpu()
    x_reduced_implicit = x_reduced_implicit.detach().cpu()

    # ================================ plotting ================================
    plot_idx = 0
    u = u.detach().cpu()
    u_reconstructed_full = u_reconstructed_full.detach().cpu()
    u_reconstructed_truncated = u_reconstructed_truncated.detach().cpu()

    fig, ax = plt.subplots()
    ax.set_title("Losses explicit")
    ax.plot(losses_full_explicit, label="full explicit")
    ax.plot(losses_full_implicit, label="full implicit")
    ax.plot(losses_reduced_explicit, label="reduced explicit")
    ax.plot(losses_reduced_implicit, label="reduced implicit")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss, single step")
    ax.set_yscale("log")
    ax.legend()

    def plot_heatmap(name_to_data: dict, plot_idx: int, vmin, vmax):
        fig, axes = plt.subplots(len(name_to_data), 1, sharex=True, sharey=True)
        # combined = torch.cat([v[..., plot_idx] for v in name_to_data.values()])
        # vmin = combined.min()
        # vmax = combined.max()
        for ax, (name, data) in zip(axes, name_to_data.items()):
            ax.imshow(
                data[..., plot_idx], vmin=vmin, vmax=vmax, cmap="jet", origin="lower"
            )
            ax.set_title(name)
        axes[-1].set_xlabel("t")
        axes[-1].set_ylabel("x")
        return fig, axes

    vmin = u.min()
    vmax = u.max()
    fig, axes = plot_heatmap(
        {
            f"true full": u,
            f"full explicit": x_full_explicit,
            f"full implicit": x_full_implicit,
            f"true reduced N={n_modes}": u_reconstructed_truncated,
            f"reduced explicit": x_reduced_explicit,
            f"reduced implicit": x_reduced_implicit,
        },
        plot_idx,
        vmin,
        vmax,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    vmin = torch.cat((K, K_full_explicit, K_full_implicit)).min()
    vmax = torch.cat((K, K_full_explicit, K_full_implicit)).max()
    ax1.imshow(K, vmin=vmin, vmax=vmax)
    ax1.set_title("K implicit")
    ax2.imshow(K_full_explicit, vmin=vmin, vmax=vmax)
    ax2.set_title("K estimated explicit")
    ax3.imshow(K_full_implicit, vmin=vmin, vmax=vmax)
    ax3.set_title("K estimated implicit")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    vmin = torch.cat((K_reduced, K_reduced_explicit, K_reduced_implicit)).min()
    vmax = torch.cat((K_reduced, K_reduced_explicit, K_reduced_implicit)).max()
    ax1.imshow(K_reduced, vmin=vmin, vmax=vmax)
    ax1.set_title("K reduced implicit")
    ax2.imshow(K_reduced_explicit, vmin=vmin, vmax=vmax)
    ax2.set_title("K reduced estimated explicit")
    ax3.imshow(K_reduced_implicit, vmin=vmin, vmax=vmax)
    ax3.set_title("K reduced estimated implicit")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(torch.linalg.eigvals(K).abs().sort(descending=True)[0])
    ax1.set_title("K full eigenvalues")
    ax2.plot(torch.linalg.eigvals(K_full_explicit).abs().sort(descending=True)[0])
    ax2.set_title("K full explicit eigenvalues")
    ax3.plot(torch.linalg.eigvals(K_full_implicit).abs().sort(descending=True)[0])
    ax3.set_title("K full implicit eigenvalues")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(torch.linalg.eigvals(K_reduced).abs().sort(descending=True)[0])
    ax1.set_title("K reduced eigenvalues")
    ax2.plot(torch.linalg.eigvals(K_reduced_explicit).abs().sort(descending=True)[0])
    ax2.set_title("K reduced explicit eigenvalues")
    ax3.plot(torch.linalg.eigvals(K_reduced_implicit).abs().sort(descending=True)[0])
    ax3.set_title("K full implicit eigenvalues")

    plt.show()
