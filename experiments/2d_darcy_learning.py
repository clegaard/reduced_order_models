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


def train(pressures, rhs, n_iterations):

    K = torch.randn(
        pressures.shape[0],
        pressures.shape[0],
        dtype=pressures.dtype,
        device=pressures.device,
        requires_grad=True,
    )
    xavier_uniform_(K)
    optimizer = torch.optim.Adam([K], lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min")
    losses = []
    for _ in tqdm(range(n_iterations), desc="training step"):
        pressures_estimated = torch.linalg.solve(K, rhs)
        loss = F.mse_loss(pressures_estimated, pressures)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)
        losses.append(loss.item())

    return K, losses


# def solve(points, Ks, rhs, permeabilities):
#     pressures = torch.zeros_like(points)
#     K_total = torch.zeros_like(points.shape[0], points.shape[0])

#     for K, perm in zip(Ks, permeabilities):
#         K_total += K * perm

#     pressures = torch.linalg.solve(K_total, rhs)

#     return pressures

if __name__ == "__main__":

    n_training_iterations = 100000

    # =================== load data ===================

    mat = scipy.io.loadmat("matlab_files/pressure.mat")
    p_coarse = torch.tensor(mat["Pressure_coarse"])
    xy_coarse = torch.tensor(mat["CCoord"])
    rhs_coarse = torch.tensor(mat["rhs_coarse"])
    rhs_fine = torch.tensor(mat["rhs_fine"])

    p_fine = torch.tensor(mat["Pressure_fine"])
    xy_fine = torch.tensor(mat["FCoord"])

    # =================== train ===================

    K, losses = train(p_coarse, rhs_coarse, n_training_iterations)
    p_coarse_estimated = torch.linalg.solve(K, rhs_coarse).detach().cpu()
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
