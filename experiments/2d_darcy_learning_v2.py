import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.linalg
from torch.nn.init import xavier_uniform_
import numpy as np
import torch.nn.functional as F
from random import randint
from pyDOE import lhs
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger


class LinearDarcyBlock(pl.LightningModule):
    """Full dimensional BlockDarcy solver, estimating the K matrix that solves the problem: K p = f"""

    def __init__(self, n_nodes, n_blocks):
        super().__init__()

        self.B = nn.parameter.Parameter(
            torch.empty(
                (n_blocks, n_nodes, n_nodes),
                requires_grad=True,
                device=self.device,
                dtype=self.dtype,
            )
        )
        xavier_uniform_(self.B)

    def forward(self, rhs, permeabilities):
        return torch.linalg.solve(self._get_K_total(permeabilities), rhs)

    def training_step(self, batch, batch_idx):
        pressures, permeabilities, rhs = batch

        rhs_estimated = (
            self._get_K_total(permeabilities) @ pressures.unsqueeze(-1)
        ).squeeze(-1)
        loss = F.mse_loss(rhs_estimated, rhs)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pressures, permeabilities, rhs = batch

        rhs_estimated = (
            self._get_K_total(permeabilities) @ pressures.unsqueeze(-1)
        ).squeeze(-1)
        loss = F.mse_loss(rhs_estimated, rhs)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _get_K_total(self, permeabilities):
        K = self.get_K_matrices()
        K = (K.T @ permeabilities.T).T
        return K

    def get_K_matrices(self):
        return self.B.transpose(1, 2) @ self.B

    def training_step(self, batch, batch_idx):
        pressures, permeabilities, rhs = batch

        rhs_estimated = (
            self._get_K_total(permeabilities) @ pressures.unsqueeze(-1)
        ).squeeze(-1)
        loss = F.mse_loss(rhs_estimated, rhs)

        self.log("train_loss", loss)
        return loss


class ReducedLinearDarcyBlock(LinearDarcyBlock):
    """Reduced dimensional BlockDarcy solver, estimating the K matrix that solves the problem: Φ'KΦ Φ'p = Φ'f"""

    def __init__(self, projection_matrix, n_blocks):

        super().__init__(projection_matrix.shape[0], n_blocks)

        self.projection_matrix = nn.Parameter(
            torch.tensor(projection_matrix, device=self.device, dtype=self.dtype),
            requires_grad=False,
        )

    def forward(self, rhs, permeabilities):

        rhs_reduced = rhs @ self.projection_matrix.T

        pressures_reduced = super().forward(rhs_reduced, permeabilities)

        pressures = pressures_reduced @ self.projection_matrix
        return pressures

    def training_step(self, batch, batch_idx):
        pressures, permeabilities, rhs = batch

        pressures_reduced = pressures @ self.projection_matrix.T
        rhs_reduced = rhs @ self.projection_matrix.T
        batch_reduced = (pressures_reduced, permeabilities, rhs_reduced)

        return super().training_step(batch_reduced, batch_idx)

    def validation_step(self, batch, batch_idx):
        pressures, permeabilities, rhs = batch

        pressures_reduced = pressures @ self.projection_matrix.T
        rhs_reduced = rhs @ self.projection_matrix.T
        batch_reduced = (pressures_reduced, permeabilities, rhs_reduced)

        return super().validation_step(batch_reduced, batch_idx)


def darcy_block_solver(Ks, permeabilities, rhs):
    K = torch.stack(Ks, dim=-1)
    K = K.to_dense()
    K = K.unsqueeze(-2) * permeabilities
    K = K.sum(dim=-1).T
    rhss = rhs.squeeze(-1)
    pressures = torch.linalg.solve(K, rhss)

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

    max_epochs = 10000
    batch_size = 75
    # n_experiments = 25
    n_train = 75
    n_validate = 25
    n_plot = 10
    n_blocks = 9

    n_modes = 9  # set to None for no reduction

    if n_modes is not None:
        assert (
            n_train >= n_modes
        ), "the number of modes can not exceed the number of experiment"

    # =================== load data ===================

    mat = scipy.io.loadmat("matlab_files/pressure.mat")
    xy_coords = torch.tensor(mat["CCoord"])
    rhs = torch.tensor(mat["rhs_coarse"]).squeeze(-1)
    Ks = [scipy_sparse_to_torch_sparse(k[0]) for k in mat["Kc"]]

    # ============ generate data ===================

    permeabilities_train = torch.tensor(
        lhs(
            n=n_blocks,
            samples=n_train,
        )
    )
    _, pressures_train = darcy_block_solver(Ks, permeabilities_train, rhs)

    permeabilities_validate = torch.tensor(
        lhs(
            n=n_blocks,
            samples=n_validate,
        )
    )
    _, pressures_validate = darcy_block_solver(Ks, permeabilities_validate, rhs)

    # =================== dimensionality reduction =============
    U, S, V = torch.linalg.svd(pressures_train, full_matrices=False)
    V = V[:n_modes]
    # model = LinearDarcyBlock(
    #     n_nodes=pressures_train.shape[1], n_blocks=n_blocks
    # ).double()

    model = ReducedLinearDarcyBlock(V, n_blocks=n_blocks).double()

    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)

    rhs_train = rhs.T.repeat(n_train, 1)
    rhs_validate = rhs.T.repeat(n_validate, 1)
    train_data = TensorDataset(pressures_train, permeabilities_train, rhs_train)
    validate_data = TensorDataset(
        pressures_validate, permeabilities_validate, rhs_validate
    )
    train_loader = DataLoader(train_data, batch_size=batch_size)
    validate_loader = DataLoader(validate_data, batch_size=batch_size)

    logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer.fit(model, train_loader, validate_loader)

    model = model.cpu()
    K = model.get_K_matrices().detach()
    p_estimated = model(rhs_validate, permeabilities_validate).detach()

    # ================== add outlet to solution =======================
    p_with_outlet = add_outlet_to_solution(pressures_validate.T, xy_coords)
    p_estimated_with_outlet = add_outlet_to_solution(p_estimated.T, xy_coords)

    # =================== plotting ===================

    n_plot = min(n_validate, n_plot)

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
        fig.canvas.manager.set_window_title(f"i: {idx} a={permeabilities_train[idx]}")

    ids = [randint(0, n_validate - 1) for _ in range(n_plot)]
    for i in ids:
        plot_3d(i)

    # ---------------- plot weights ------------------
    fig, axes = plt.subplots(1, n_blocks)
    for ax, k in zip(axes, K):
        ax.imshow(k)

    # ---------------- plot singular values ------------------
    # if n_modes is not None:
    #     fig, ax = plt.subplots()
    #     # ax.stem(S)
    #     torch.sum(S)
    #     ax.stem(S / torch.sum(S))

    #     # ax.axvline(n_modes + 0.5, label="number of modes", c="green")
    #     # ax.set_yscale("log")
    #     plt.legend()

    plt.show()
