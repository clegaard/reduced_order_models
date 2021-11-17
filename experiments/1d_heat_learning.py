import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_

if __name__ == "__main__":

    n_train_iterations = 10000
    n_train_steps = 149
    n_modes = 10
    device = "cuda"

    Δt = 0.01
    u, t, x, M = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=1.49,
        Δt=Δt,
        x_start=0.0,
        x_end=1.0,
        Δx=0.01,
        α=-0.5,
    )

    assert n_train_steps <= len(
        t
    ), f"training data must have a length greater or equal than the number of training steps, actual is {len(t)} and {n_train_steps}"

    # TODO
    u = u.T
    # end TODO

    u = torch.tensor(u, dtype=torch.float32).to(device)
    u = u[:, : n_train_steps + 1]
    t = t[: n_train_steps + 1]
    u0 = u[:, 0].to(device)

    # ================== Dimensionality Reduction ======================
    U, S, V = torch.linalg.svd(u, full_matrices=False)

    z = U.T @ u
    u_reconstructed_full = U @ z
    Ut = U[:, :n_modes]
    zt = Ut.T @ u
    u_reconstructed_truncated = Ut @ zt

    # ========================== Training =============================

    results = {
        "full": {
            "explicit": {"losses": [], "sim": None, "step": None, "opt": None},
            "implicit": {"losses": [], "sim": None, "step": None, "opt": None},
        },
        "reduced": {"explicit": {"losses": [], "sim": None, "step": None, "opt": None}},
    }

    M_explicit = torch.nn.Linear(M.shape[0], M.shape[0], bias=False).to(device)
    M_implicit = torch.nn.Linear(M.shape[0], M.shape[0], bias=False).to(device)
    M_reduced_explicit = torch.nn.Linear(n_modes, n_modes, bias=False).to(device)

    xavier_uniform_(M_explicit.weight)
    xavier_uniform_(M_implicit.weight)
    xavier_uniform_(M_reduced_explicit.weight)
    results["full"]["explicit"]["opt"] = torch.optim.Adam(M_explicit.parameters())
    results["full"]["implicit"]["opt"] = torch.optim.Adam(M_implicit.parameters())
    results["reduced"]["explicit"]["opt"] = torch.optim.Adam(
        M_reduced_explicit.parameters()
    )

    # =========== full system -> forward euler ===========

    for _ in tqdm(range(n_train_iterations), desc="training explicit"):
        u_next = u + M_explicit(u.T).T * Δt
        loss = F.mse_loss(u_next[:, :-1], u[:, 1:])
        loss.backward()
        results["full"]["explicit"]["losses"].append(loss.item())
        results["full"]["explicit"]["opt"].step()
        results["full"]["explicit"]["opt"].zero_grad()

    results["full"]["explicit"]["step"] = torch.concat(
        (u[:, :1], u_next[:, :-1]), dim=1
    )
    cur = u[:, :1]
    next = None
    us = [cur]
    for _ in t:
        next = cur + M_explicit(cur.T).T * Δt
        us.append(next)
        cur = next
    results["full"]["explicit"]["sim"] = torch.concat(us, dim=1)

    assert (
        results["full"]["explicit"]["step"].shape
        == results["full"]["explicit"]["sim"].shape
    )

    results["full"]["explicit"]["step"] = (
        results["full"]["explicit"]["step"].detach().cpu().numpy()
    )
    results["full"]["explicit"]["sim"] = (
        results["full"]["explicit"]["sim"].detach().cpu().numpy()
    )

    # =========== full system -> backward euler ===========

    for _ in tqdm(range(n_train_iterations), desc="training implicit"):
        u_prev = u + M_implicit(u.T).T * Δt
        loss = F.mse_loss(u_prev[:, 1:], u[:, :-1])
        loss.backward()
        results["full"]["implicit"]["losses"].append(loss.item())
        results["full"]["implicit"]["opt"].step()
        results["full"]["implicit"]["opt"].zero_grad()

    results["full"]["implicit"]["step"] = torch.concat(
        (u_prev[:, 1:], u[:, -1:]), dim=1
    )

    M_implicit_inv = torch.linalg.inv(
        Δt * M_implicit.weight + torch.eye(M_implicit.weight.shape[0]).to(device)
    )
    cur = u[:, :1]
    next = None
    us = [cur]
    for _ in t:
        next = M_implicit_inv @ cur
        us.append(next)
        cur = next
    results["full"]["implicit"]["sim"] = torch.concat(us, dim=1)

    assert (
        results["full"]["implicit"]["step"].shape
        == results["full"]["implicit"]["sim"].shape
    )

    results["full"]["implicit"]["step"] = (
        results["full"]["implicit"]["step"].detach().cpu().numpy()
    )
    results["full"]["implicit"]["sim"] = (
        results["full"]["implicit"]["sim"].detach().cpu().numpy()
    )

    # =========== reduced system -> forward euler ===========

    for _ in tqdm(range(n_train_iterations), desc="training reduced explicit"):
        z_next = zt + M_reduced_explicit(zt.T).T * Δt
        loss = F.mse_loss(z_next[:, :-1], zt[:, 1:])
        loss.backward()
        results["reduced"]["explicit"]["losses"].append(loss.item())
        results["reduced"]["explicit"]["opt"].step()
        results["reduced"]["explicit"]["opt"].zero_grad()

    z_step = torch.concat((zt[:, :1], z_next[:, :-1]), dim=1)
    results["reduced"]["explicit"]["step"] = Ut @ z_step
    cur = zt[:, :1]
    next = None
    zs = [cur]
    for _ in t:
        next = cur + M_reduced_explicit(cur.T).T * Δt
        zs.append(next)
        cur = next

    zs = torch.concat(zs, dim=1)
    results["reduced"]["explicit"]["sim"] = Ut @ zs

    assert (
        results["reduced"]["explicit"]["step"].shape
        == results["reduced"]["explicit"]["sim"].shape
    )

    results["reduced"]["explicit"]["step"] = (
        results["reduced"]["explicit"]["step"].detach().cpu().numpy()
    )
    results["reduced"]["explicit"]["sim"] = (
        results["reduced"]["explicit"]["sim"].detach().cpu().numpy()
    )

    # ================================ plotting ================================

    u = u.detach().cpu().numpy()
    u_reconstructed_full = u_reconstructed_full.detach().cpu().numpy()
    u_reconstructed_truncated = u_reconstructed_truncated.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.set_title("Losses explicit")
    ax.plot(results["full"]["explicit"]["losses"], label="full explicit")
    ax.plot(results["full"]["implicit"]["losses"], label="full implicit")
    ax.plot(results["reduced"]["explicit"]["losses"], label="full reduced")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss, single step")
    ax.legend()

    fig, ax = heatmap_1d(u.T, x, t)
    fig.canvas.manager.set_window_title(f"ground truth")

    fig, ax = heatmap_1d(u_reconstructed_full.T, x, t)
    fig.canvas.manager.set_window_title(f"ground truth reconstructed full")

    fig, ax = heatmap_1d(u_reconstructed_truncated.T, x, t)
    fig.canvas.manager.set_window_title(
        f"ground truth reconstructed truncated, N = {n_modes}"
    )

    # fig, ax = heatmap_1d(results["full"]["explicit"]["step"].T, x, t)
    # fig.canvas.manager.set_window_title(f"predicted explicit step")

    fig, ax = heatmap_1d(results["full"]["explicit"]["sim"].T, x, t)
    fig.canvas.manager.set_window_title(f"predicted explicit sim")

    # fig, ax = heatmap_1d(results["full"]["implicit"]["step"].T, x, t)
    # fig.canvas.manager.set_window_title(f"predicted implicit step")

    fig, ax = heatmap_1d(results["full"]["implicit"]["sim"].T, x, t)
    fig.canvas.manager.set_window_title(f"predicted implicit sim")

    # fig, ax = heatmap_1d(results["reduced"]["explicit"]["step"].T, x, t)
    # fig.canvas.manager.set_window_title(f"predicted reduced explicit step")

    fig, ax = heatmap_1d(results["reduced"]["explicit"]["sim"].T, x, t)
    fig.canvas.manager.set_window_title(
        f"predicted reduced explicit sim, N = {n_modes}"
    )

    plt.show()
