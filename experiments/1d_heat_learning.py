import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_

if __name__ == "__main__":

    n_train_iterations = 30000
    n_modes = 10
    device = "cuda"
    dtype = torch.float64

    Δt = 0.01
    α = 0.005
    Δx = 0.01
    lr = 1e-3
    γ = α * Δt / Δx ** 2
    u, t, x, K = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=2.0,
        Δt=Δt,
        x_start=0.0,
        x_end=1.0,
        Δx=Δx,
        α=α,
    )

    # ===================== Generate Data ==============================

    #

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

    K_reduced = Ut.T.cpu() @ K @ Ut.cpu()

    # ========================== Training =============================

    results = {
        "full": {
            "explicit": {"losses": [], "sim": None, "step": None},
            "implicit": {"losses": [], "sim": None, "step": None},
        },
        "reduced": {
            "explicit": {"losses": [], "sim": None, "step": None},
            "implicit": {"losses": [], "sim": None, "step": None},
        },
    }

    # =========== full system -> forward euler ===========

    K_explicit = torch.empty(K.shape, dtype=dtype, device=device, requires_grad=True)
    xavier_uniform_(K_explicit)
    opt = torch.optim.Adam([K_explicit], lr=lr)

    for _ in tqdm(range(n_train_iterations), desc="training explicit"):
        u_next = u + K_explicit @ u * γ
        loss = F.mse_loss(u_next[:, :-1], u[:, 1:])
        loss.backward()
        results["full"]["explicit"]["losses"].append(loss.item())
        opt.step()
        opt.zero_grad()

    results["full"]["explicit"]["step"] = torch.concat(
        (u[:, :1], u_next[:, :-1]), dim=1
    )
    cur = u[:, :1]
    next = None
    us = [cur]
    for _ in t:
        next = cur + K_explicit @ cur * γ
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
    K_explicit = K_explicit.detach().cpu().numpy()

    # =========== full system -> backward euler ===========

    K_implicit = torch.empty(K.shape, dtype=dtype, device=device, requires_grad=True)
    xavier_uniform_(K_implicit)
    opt = torch.optim.Adam([K_implicit], lr=lr)

    for _ in tqdm(range(n_train_iterations), desc="training implicit"):
        u_prev = u - K_implicit @ u * γ
        loss = F.mse_loss(u_prev[:, 1:], u[:, :-1])
        loss.backward()
        results["full"]["implicit"]["losses"].append(loss.item())
        opt.step()
        opt.zero_grad()

    results["full"]["implicit"]["step"] = torch.concat(
        (u_prev[:, 1:], u[:, -1:]), dim=1
    )

    M_implicit_inv = torch.linalg.inv(
        torch.eye(K_implicit.shape[0]).to(device) - K_implicit * γ
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

    K_implicit = K_implicit.detach().cpu().numpy()

    # =========== reduced system -> forward euler ===========

    K_reduced_explicit = torch.empty(
        (n_modes, n_modes), dtype=dtype, device=device, requires_grad=True
    )
    xavier_uniform_(K_reduced_explicit)
    opt = torch.optim.Adam([K_reduced_explicit], lr=lr)

    for _ in tqdm(range(n_train_iterations), desc="training reduced explicit"):
        z_next = zt + K_reduced_explicit @ zt * γ
        loss = F.mse_loss(z_next[:, :-1], zt[:, 1:])
        loss.backward()
        results["reduced"]["explicit"]["losses"].append(loss.item())
        opt.step()
        opt.zero_grad()

    zs_step = torch.concat((zt[:, :1], z_next[:, :-1]), dim=1)
    results["reduced"]["explicit"]["step"] = Ut @ zs_step
    cur = zt[:, :1]
    next = None
    zs = [cur]
    for _ in t:
        next = cur + K_reduced_explicit @ cur * γ
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
    K_reduced_explicit = K_reduced_explicit.detach().cpu().numpy()

    # ============= Reduced system -> Backward euler =================

    K_reduced_implicit = torch.empty(
        (n_modes, n_modes), dtype=dtype, device=device, requires_grad=True
    )
    xavier_uniform_(K_reduced_implicit)
    opt = torch.optim.Adam([K_reduced_implicit], lr=lr)

    for _ in tqdm(range(n_train_iterations), desc="training reduced implicit"):
        z_prev = zt - K_reduced_implicit @ zt * γ
        loss = F.mse_loss(z_prev[:, 1:], zt[:, :-1])
        loss.backward()
        results["reduced"]["implicit"]["losses"].append(loss.item())
        opt.step()
        opt.zero_grad()

    zs_step = torch.concat((z_prev[:, 1:], zt[:, -1:]), dim=1)
    results["reduced"]["implicit"]["step"] = Ut @ zs_step

    M_reduced_implicit_inv = torch.linalg.inv(
        torch.eye(K_reduced_implicit.shape[0]).to(device) - γ * K_reduced_implicit
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

    K_reduced_implicit = K_reduced_implicit.detach().cpu().numpy()

    # ================================ plotting ================================

    u = u.detach().cpu().numpy()
    u_reconstructed_full = u_reconstructed_full.detach().cpu().numpy()
    u_reconstructed_truncated = u_reconstructed_truncated.detach().cpu().numpy()

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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(K)
    ax1.set_title("K implicit")
    ax2.imshow(K_explicit)
    ax3.set_title("K estimated explicit")
    ax3.imshow(K_implicit)
    ax2.set_title("K estimated implicit")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(K_reduced)
    ax1.set_title("K reduced implicit")
    ax2.imshow(K_reduced_explicit)
    ax3.set_title("K reduced estimated explicit")
    ax3.imshow(K_reduced_implicit)
    ax2.set_title("K reduced estimated implicit")

    plt.show()
