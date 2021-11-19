import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_

if __name__ == "__main__":

    n_train_iterations = 1000
    n_modes = 10
    device = "cuda"
    dtype = torch.float64

    Δt = 0.01
    α = 0.005
    Δx = 0.01
    lr = 1e-3
    γ = 0.01  # α * Δt / Δx ** 2
    u, t, x, M = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=2.0,
        Δt=Δt,
        x_start=0.0,
        x_end=1.0,
        Δx=Δx,
        α=α,
    )

    # TODO
    u = u.T
    # end TODO

    u = torch.tensor(u, dtype=dtype).to(device)

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
        "reduced": {
            "explicit": {"losses": [], "sim": None, "step": None, "opt": None},
            "implicit": {"losses": [], "sim": None, "step": None, "opt": None},
        },
    }

    K_explicit = torch.nn.Linear(
        M.shape[0], M.shape[0], bias=False, dtype=dtype, device=device
    )
    K_implicit = torch.nn.Linear(
        M.shape[0], M.shape[0], bias=False, dtype=dtype, device=device
    )
    M_reduced_explicit = torch.nn.Linear(
        n_modes, n_modes, bias=False, dtype=dtype, device=device
    )
    M_reduced_implicit = torch.nn.Linear(
        n_modes, n_modes, bias=False, dtype=dtype, device=device
    )

    xavier_uniform_(K_explicit.weight)
    xavier_uniform_(K_implicit.weight)
    xavier_uniform_(M_reduced_explicit.weight)
    xavier_uniform_(M_reduced_implicit.weight)
    results["full"]["explicit"]["opt"] = torch.optim.Adam(
        K_explicit.parameters(), lr=lr
    )
    results["full"]["implicit"]["opt"] = torch.optim.Adam(
        K_implicit.parameters(), lr=lr
    )
    results["reduced"]["explicit"]["opt"] = torch.optim.Adam(
        M_reduced_explicit.parameters(), lr=lr
    )
    results["reduced"]["implicit"]["opt"] = torch.optim.Adam(
        M_reduced_implicit.parameters(), lr=lr
    )

    # =========== full system -> forward euler ===========

    for _ in tqdm(range(n_train_iterations), desc="training explicit"):
        u_next = u + K_explicit(u.T).T * γ
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
        next = cur + K_explicit(cur.T).T * γ
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
        u_prev = u - K_implicit(u.T).T * γ
        loss = F.mse_loss(u_prev[:, 1:], u[:, :-1])
        loss.backward()
        results["full"]["implicit"]["losses"].append(loss.item())
        results["full"]["implicit"]["opt"].step()
        results["full"]["implicit"]["opt"].zero_grad()

    results["full"]["implicit"]["step"] = torch.concat(
        (u_prev[:, 1:], u[:, -1:]), dim=1
    )

    M_implicit_inv = torch.linalg.inv(
        torch.eye(K_implicit.weight.shape[0]).to(device) - K_implicit.weight * γ
    )
    cond = torch.linalg.cond(M_implicit_inv)
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
        z_next = zt + M_reduced_explicit(zt.T).T * γ
        loss = F.mse_loss(z_next[:, :-1], zt[:, 1:])
        loss.backward()
        results["reduced"]["explicit"]["losses"].append(loss.item())
        results["reduced"]["explicit"]["opt"].step()
        results["reduced"]["explicit"]["opt"].zero_grad()

    zs_step = torch.concat((zt[:, :1], z_next[:, :-1]), dim=1)
    results["reduced"]["explicit"]["step"] = Ut @ zs_step
    cur = zt[:, :1]
    next = None
    zs = [cur]
    for _ in t:
        next = cur + M_reduced_explicit(cur.T).T * γ
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

    # ============= Reduced system -> Backward euler =================
    for _ in tqdm(range(n_train_iterations), desc="training reduced implicit"):
        z_prev = zt - M_reduced_implicit(zt.T).T * γ
        loss = F.mse_loss(z_prev[:, 1:], zt[:, :-1])
        loss.backward()
        results["reduced"]["implicit"]["losses"].append(loss.item())
        results["reduced"]["implicit"]["opt"].step()
        results["reduced"]["implicit"]["opt"].zero_grad()

    zs_step = torch.concat((z_prev[:, 1:], zt[:, -1:]), dim=1)
    results["reduced"]["implicit"]["step"] = Ut @ zs_step

    M_reduced_implicit_inv = torch.linalg.inv(
        torch.eye(M_reduced_implicit.weight.shape[0]).to(device)
        - γ * M_reduced_implicit.weight
    )
    cur = zt[:, :1]
    next = None
    zs = [cur]
    for _ in t:
        next = M_reduced_implicit_inv @ cur
        zs.append(next)
        cur = next
    zs = torch.concat(zs, dim=1)
    results["reduced"]["implicit"]["sim"] = Ut @ zs

    assert (
        results["reduced"]["implicit"]["step"].shape
        == results["reduced"]["implicit"]["sim"].shape
    )

    results["reduced"]["implicit"]["step"] = (
        results["reduced"]["implicit"]["step"].detach().cpu().numpy()
    )
    results["reduced"]["implicit"]["sim"] = (
        results["reduced"]["implicit"]["sim"].detach().cpu().numpy()
    )

    # ================================ plotting ================================

    u = u.detach().cpu().numpy()
    u_reconstructed_full = u_reconstructed_full.detach().cpu().numpy()
    u_reconstructed_truncated = u_reconstructed_truncated.detach().cpu().numpy()
    K_implicit = K_implicit.weight.detach().cpu().numpy()
    M_reduced_explicit = M_reduced_explicit.weight.detach().cpu().numpy()
    M_reduced_implicit = M_reduced_implicit.weight.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.set_title("Losses explicit")
    ax.plot(results["full"]["explicit"]["losses"], label="full explicit")
    ax.plot(results["full"]["implicit"]["losses"], label="full implicit")
    ax.plot(results["reduced"]["explicit"]["losses"], label="reduced explicit")
    ax.plot(results["reduced"]["implicit"]["losses"], label="reduced implicit")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss, single step")
    ax.set_yscale("log")
    ax.legend()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    ax1.imshow(u, cmap="jet", origin="lower")
    ax1.set_title("true")
    ax2.imshow(results["full"]["explicit"]["sim"], cmap="jet", origin="lower")
    ax2.set_title("explicit")
    im = ax3.imshow(results["full"]["implicit"]["sim"], cmap="jet", origin="lower")
    ax3.set_title("implicit ")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    # plt.colorbar(im)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    ax1.imshow(u_reconstructed_truncated, cmap="jet", origin="lower")
    ax1.set_title(f"true, reduced N={n_modes}")
    ax2.imshow(results["reduced"]["explicit"]["sim"], cmap="jet", origin="lower")
    ax2.set_title("explicit")
    im = ax3.imshow(results["reduced"]["implicit"]["sim"], cmap="jet", origin="lower")
    ax3.set_title("implicit")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    # plt.colorbar(im)

    plt.show()
