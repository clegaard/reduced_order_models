import scipy.io
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
import torch.linalg
from torch.nn.init import xavier_uniform_
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_model(hc):
    pass


# def assemble_k_matrices(points, connectivity)

import torch.nn.functional as F

from tqdm import tqdm


def train_seperable(pressures, rhs, n_iterations, estimation_mode, permeabilities, Ks):
    # def make_K():
    #     K = torch.randn(
    #         pressures.shape[0],
    #         pressures.shape[0],
    #         dtype=pressures.dtype,
    #         device=pressures.device,
    #         requires_grad=True,
    #     )
    #     xavier_uniform_(K)
    #     return K

    # Ks = [make_K() for _ in permeabilities]

    # optimizer = torch.optim.Adam(Ks, lr=1e-4)
    losses = []
    for _ in tqdm(range(n_iterations), desc="training step"):

        if estimation_mode == "lhs":
            K_total = torch.sum(torch.stack(Ks, dim=-1) * permeabilities, dim=-1)
            pressures_estimated = torch.linalg.solve(K_total, rhs)
        elif estimation_mode == "rhs":
            pass
        else:
            raise ValueError("Unrecognized estimation mode")

        loss = F.mse_loss(pressures_estimated, pressures)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        losses.append(loss.item())

    return K_total, losses


def train(pressures, rhs, n_iterations, estimation_mode):

    K = torch.randn(
        pressures.shape[0],
        pressures.shape[0],
        dtype=pressures.dtype,
        device=pressures.device,
        requires_grad=True,
    )
    xavier_uniform_(K)
    optimizer = torch.optim.Adam([K], lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=100)
    losses = []
    for iter in tqdm(range(n_iterations), desc="training step"):

        if estimation_mode == "rhs":
            pressures_estimated = K @ rhs
        elif estimation_mode == "lhs":
            pressures_estimated = torch.linalg.solve(K, rhs)
        else:
            raise ValueError("Unrecognized estimation mode")

        loss = F.mse_loss(pressures_estimated, pressures)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iter > 10000:
            scheduler.step(loss)
        losses.append(loss.item())

    return K, losses


if __name__ == "__main__":

    n_training_iterations = 10000
    estimation_mode = "rhs"

    # =================== load data ===================

    mat = scipy.io.loadmat("matlab_files/pressure.mat")
    permeabilities = torch.tensor(mat["permeability"]).reshape(-1)
    p_coarse = torch.tensor(mat["Pressure_coarse"])
    xy_coarse = torch.tensor(mat["CCoord"])
    rhs_coarse = torch.tensor(mat["rhs_coarse"])
    rhs_fine = torch.tensor(mat["rhs_fine"])
    Kc = [torch.tensor(k[0]) for k in mat["Kc"]]
    Kf = [torch.tensor(k[0]) for k in mat["Kf"]]

    p_fine = torch.tensor(mat["Pressure_fine"])
    xy_fine = torch.tensor(mat["FCoord"])

    # TODO
    # K_total = torch.zeros_like(Kc[0])
    # for k, p in zip(Kc, permeabilities):
    #     K_total += k * p
    # plt.imshow(K_total)
    # plt.show()
    # pressures_estimated = torch.linalg.solve(
    #     K_total, rhs_coarse
    # )  # impossible to invert since diagonal is 0

    # =================== train ===================

    K, losses = train(
        p_coarse, rhs_coarse, n_training_iterations, estimation_mode=estimation_mode
    )
    # K, losses = train_seperable(
    #     p_coarse, rhs_coarse, n_training_iterations, "lhs", permeabilities, Kc
    # )

    if estimation_mode == "rhs":
        p_coarse_estimated = (K @ rhs_coarse).detach().cpu()
    elif estimation_mode == "lhs":
        p_coarse_estimated = torch.linalg.solve(K, rhs_coarse).detach().cpu()
    else:
        raise ValueError()

    K = K.detach().cpu()

    # =================== plotting ===================

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection="3d")

    x = xy_coarse[:, 0]
    y = xy_coarse[:, 1]
    z = p_coarse[:, 0]

    ax.plot_trisurf(
        x,
        y,
        z,
        cmap=plt.cm.Spectral,
    )
    ax.set_title("coarse")

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    x = xy_fine[:, 0]
    y = xy_fine[:, 1]
    z = p_fine[:, 0]
    ax.plot_trisurf(
        x,
        y,
        z,
        cmap=plt.cm.Spectral,
    )
    ax.set_title("fine")

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    x = xy_coarse[:, 0]
    y = xy_coarse[:, 1]
    z = p_coarse_estimated[:, 0]

    ax.plot_trisurf(
        x,
        y,
        z,
        cmap=plt.cm.Spectral,
    )
    ax.set_title("coarse")

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")

    fig, ax = plt.subplots()
    ax.imshow(K)

    plt.show()
