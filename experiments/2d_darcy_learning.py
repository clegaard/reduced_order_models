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
from pyDOE import lhs


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
        if iter > 10000:
            scheduler.step(loss)
        losses.append(loss.item())

    return K, losses


# function Pressure = DarcyBlockSolver(P,Permeability)

# Pressure = zeros(size(P.K{1},1),size(Permeability,1));

# for k=1:size(Permeability,1)
#      K=Permeability(k,1).*P.K{1};
#      for j= 2:9
#      K= K+Permeability(k,j).*P.K{j};
#      end
#     %rhs= Permeability(k,4)*P.rhs; % constant pressure gradient
#     rhs= P.rhs; % constant inlet velocity
#     Pressure(:,k) = K\rhs;
# end


# end


def darcy_block_solver(Ks, permeabilities, rhs):
    def batched():
        K_total = torch.stack(Ks, dim=-1)
        K_total = K_total.to_dense()
        K_total = K_total.unsqueeze(-2) * permeabilities
        K_total = K_total.sum(dim=-1)
        rhss = rhs.squeeze(-1)
        pressures = torch.linalg.solve(K_total.T, rhss).T
        return pressures

    def looped():
        pressures = []
        for ps in permeabilities:
            K_total = torch.zeros_like(Ks[0])
            for k, p in zip(Ks, ps):
                K_total += k * p
            pressures.append(torch.linalg.solve(K_total.to_dense(), rhs))

        pressures = torch.cat(pressures, dim=-1)
        return pressures

    # pressures_b = batched()
    # pressures_l = looped()
    # assert pressures_l.allclose(pressures_b)

    pressures = looped()

    return pressures


def scipy_sparse_to_torch_sparse(csr):
    import numpy as np

    coo = csr.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    # i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)

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


if __name__ == "__main__":

    n_training_iterations = 5000
    n_experiments = 1
    estimation_mode = "minimize f residual"

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
    permeabilities[0] = torch.tensor(
        [1.0, 0.0005, 1.0, 1.0, 0.0005, 1.0, 1.0, 1.0, 1.0]
    )

    pressures = darcy_block_solver(Ks, permeabilities, rhs)
    p_with_outlet = add_outlet_to_solution(pressures, xy_coords)

    # =================== train ===================

    K, losses = train(
        pressures, rhs, n_training_iterations, estimation_mode=estimation_mode
    )

    if estimation_mode == "pressure rhs":
        p_estimated = (K @ rhs).detach().cpu()
    elif estimation_mode in {"pressure lhs", "minimize f residual"}:
        p_estimated = torch.linalg.solve(K, rhs).detach().cpu()
    else:
        raise ValueError()

    p_estimated_with_outlet = add_outlet_to_solution(p_estimated, xy_coords)

    K = K.detach().cpu()

    # =================== plotting ===================

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    x = xy_coords[:, 0]
    y = xy_coords[:, 1]
    z = p_with_outlet[..., 0]

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
    z = p_estimated_with_outlet[..., 0]

    ax.plot_trisurf(
        x,
        y,
        z,
        cmap=plt.cm.Spectral,
    )
    ax.set_title("estimated")

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")

    fig, ax = plt.subplots()
    ax.imshow(K)

    plt.show()
