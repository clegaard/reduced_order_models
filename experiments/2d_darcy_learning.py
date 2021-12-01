import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.linalg
from torch.nn.init import xavier_uniform_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pyDOE import lhs


def train_seperable(
    pressures, rhs, n_iterations, blocks, estimation_mode, lr, lr_schedule
):

    K = torch.randn(
        (pressures.shape[0], pressures.shape[0], blocks),
        dtype=pressures.dtype,
        device=pressures.device,
        requires_grad=True,
    )

    xavier_uniform_(K)
    optimizer = torch.optim.Adam([K], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min")

    # rhs = rhs.repeat(1, pressures.shape[-1])
    rhs = rhs.squeeze(-1)
    losses = []

    def eval_K(permeabilities):
        K_total = K @ permeabilities.T
        pressures_estimated = (K_total.T @ rhs).T
        return pressures_estimated

    for _ in tqdm(range(n_iterations), desc="training step"):

        if estimation_mode == "seperable rhs":
            loss = F.mse_loss(eval_K(permeabilities), pressures)
        else:
            raise ValueError("Unrecognized estimation mode")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if lr_schedule:
            scheduler.step(loss)
        losses.append(loss.item())

    return K, eval_K, losses


def train(pressures, rhs, n_iterations, estimation_mode, lr, lr_schedule):

    K = torch.randn(
        pressures.shape[0],
        pressures.shape[0],
        dtype=pressures.dtype,
        device=pressures.device,
        requires_grad=True,
    )
    xavier_uniform_(K)
    optimizer = torch.optim.Adam([K], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min")

    rhs = rhs.repeat(1, pressures.shape[-1])
    losses = []
    for iter in tqdm(range(n_iterations), desc="training step"):

        if estimation_mode == "pressure rhs":
            loss = F.mse_loss(K @ rhs, pressures)

        elif estimation_mode == "pressure lhs":
            loss = F.mse_loss(torch.linalg.solve(K, rhs), pressures)

        elif estimation_mode == "minimize f residual":
            loss = F.mse_loss(K @ pressures, rhs)
        else:
            raise ValueError("Unrecognized estimation mode")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if lr_schedule:
            scheduler.step(loss)
        losses.append(loss.item())

    return K, losses


def darcy_block_solver(Ks, permeabilities, rhs):
    def batched():
        K_total = torch.stack(Ks, dim=-1)
        K_total = K_total.to_dense()
        K_total = K_total.unsqueeze(-2) * permeabilities
        K_total = K_total.sum(dim=-1)
        rhss = rhs.squeeze(-1)
        pressures = torch.linalg.solve(K_total.T, rhss).T
        return K_total, pressures

    def looped():
        pressures = []
        for ps in permeabilities:
            K_total = torch.zeros_like(Ks[0])
            for k, p in zip(Ks, ps):
                K_total += k * p
            pressures.append(torch.linalg.solve(K_total.to_dense(), rhs))

        pressures = torch.cat(pressures, dim=-1)
        return K_total, pressures

    # pressures_b = batched()
    # pressures_l = looped()
    # assert pressures_l.allclose(pressures_b)

    K, pressures = looped()

    return K, pressures


def scipy_sparse_to_torch_sparse(csr):

    coo = csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    s = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))
    return s


def find_non_outlet_idx(xy_coords):
    m = xy_coords[:, 0].max()
    ids = (xy_coords[:, 0] != m).nonzero()

    return ids


def add_outlet_to_solution(p_without_outlet, xy_coords):
    m = xy_coords[:, 0].max()
    ids = (xy_coords[:, 0] != m).nonzero().squeeze(-1)

    p_with_outlet = torch.zeros(
        (xy_coords.shape[0], p_without_outlet.shape[-1]),
        dtype=p_without_outlet.dtype,
        device=p_without_outlet.device,
    )
    p_with_outlet[ids] = p_without_outlet

    return p_with_outlet


def batch_mult(K, X):
    return torch.matmul(X[:, None, :, :].T, K.T).T.squeeze(1)


if __name__ == "__main__":

    n_training_iterations = 10000
    lr = 1e-3
    lr_schedule = True
    n_experiments = 100
    n_plot = 10

    n_modes = 9  # set to None for no reduction
    estimation_mode = "seperable rhs"

    if n_modes is not None:
        assert (
            n_experiments >= n_modes
        ), "the number of modes can not exceed the number of experiment"

    # =================== load data ===================

    mat = scipy.io.loadmat("matlab_files/pressure.mat")
    # permeabilities = torch.tensor(mat["permeability"]).reshape(-1)
    # pressures = torch.tensor(mat["Pressure_coarse"])
    xy_coords = torch.tensor(mat["CCoord"])
    rhs = torch.tensor(mat["rhs_coarse"])
    # rhs_fine = torch.tensor(mat["rhs_fine"])
    Ks = [scipy_sparse_to_torch_sparse(k[0]) for k in mat["Kc"]]
    # Kf = [torch.tensor(k[0]) for k in mat["Kf"]]

    # p_fine = torch.tensor(mat["Pressure_fine"])
    # xy_fine = torch.tensor(mat["FCoord"])

    # ============ generate data ===================

    permeabilities = torch.tensor(
        lhs(
            n=9,
            samples=n_experiments,
        )
    )
    # permeabilities[0] = torch.tensor(
    #     [1.0, 0.0005, 1.0, 1.0, 0.0005, 1.0, 1.0, 1.0, 1.0]
    # )
    n_blocks = permeabilities.shape[1]
    K, pressures = darcy_block_solver(Ks, permeabilities, rhs)

    # =================== dimensionality reduction =============
    if n_modes is not None:
        U, S, V = torch.linalg.svd(pressures, full_matrices=False)

        z = U.T @ pressures
        p_without_outlet_reconstructed_full = U @ z

        Ut = U[:, :n_modes]
        zt = Ut.T @ pressures
        p_reconstructed_truncated = Ut @ zt
        K_reduced = Ut.T.cpu() @ K.to_dense() @ Ut.cpu()

        p_train = zt
        rhs_train = Ut.T @ rhs
    else:
        p_train = pressures
        rhs_train = rhs

    # =================== train ===================

    # K, losses = train(
    #     p_train,
    #     rhs_train,
    #     n_training_iterations,
    #     estimation_mode=estimation_mode,
    #     lr=lr,
    #     lr_schedule=lr_schedule,
    # )

    K, K_eval, losses = train_seperable(
        p_train,
        rhs_train,
        n_training_iterations,
        blocks=n_blocks,
        estimation_mode=estimation_mode,
        lr=lr,
        lr_schedule=lr_schedule,
    )

    if estimation_mode == "pressure rhs":
        p_estimated = (K @ rhs_train).detach().cpu()
    elif estimation_mode in {"pressure lhs", "minimize f residual"}:
        p_estimated = torch.linalg.solve(K, rhs_train).detach().cpu()
    elif estimation_mode == "seperable rhs":
        p_estimated = K_eval(permeabilities).detach().cpu()
    else:
        raise ValueError()

    # ================== project solution to full order space ==================

    if n_modes is not None:
        p_estimated = Ut @ p_estimated

    # ================== add outlet to solution =======================
    p_estimated_with_outlet = add_outlet_to_solution(p_estimated, xy_coords)
    p_with_outlet = add_outlet_to_solution(pressures, xy_coords)
    K = K.detach().cpu()

    # =================== plotting ===================

    n_plot = min(n_experiments, n_plot)

    def plot_3d(idx):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        x = xy_coords[:, 0]
        y = xy_coords[:, 1]
        z = p_with_outlet[..., idx]

        ax.plot_trisurf(
            x,
            y,
            z,
            cmap=plt.cm.Spectral,
        )
        ax.set_title("true")

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        x = xy_coords[:, 0]
        y = xy_coords[:, 1]
        z = p_estimated_with_outlet[..., idx]

        ax.plot_trisurf(
            x,
            y,
            z,
            cmap=plt.cm.Spectral,
        )
        ax.set_title("estimated")
        fig.canvas.manager.set_window_title(f"i: {idx} a={permeabilities[idx]}")

    for i in range(n_plot):
        plot_3d(i)

    # ---------------- plot loss ------------------
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")

    # ---------------- plot weights ------------------
    fig, axes = plt.subplots(1, n_blocks)
    for ax, k in zip(axes, K.T):
        ax.imshow(k)

    # ---------------- plot singular values ------------------
    if n_modes is not None:
        fig, ax = plt.subplots()
        ax.plot(S)
        ax.axvline(n_modes, label="number of modes", c="red")
        plt.legend()

    plt.show()
