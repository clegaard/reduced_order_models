import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.animation import FuncAnimation, PillowWriter
from math import ceil
import numpy as np
from os import makedirs


def f(x, e=3):
    return x ** e * (x ** 2 - 1)


# def g(x):
#     return x ** 9 * (156 * x ** 2 - 110)


# def f(x):
#     return torch.sin(x * 31)


if __name__ == "__main__":

    hidden_dim = 128

    dx_dense = 0.001
    dx_collocation = 0.1
    device = "cuda"
    max_epochs = 5000
    animate_every = ceil(max_epochs / 100)

    model = nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Softplus(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    x_dense = torch.arange(-1, 1 + dx_dense, dx_dense).unsqueeze(-1).to(device)
    x_collocation = torch.arange(
        -1,
        1 + dx_collocation,
        dx_collocation,
        device=device,
    ).unsqueeze(-1)
    x_collocation.requires_grad = True
    x_collocation_original = x_collocation.clone().detach()

    noise = 0.0  # 0.001

    # ================== optimizers ==================
    optimizer_min = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_max = torch.optim.Adam((x_collocation,), lr=1e-3)

    # scheduler = ReduceLROnPlateau(optimizer_min, "min", patience=100)

    losses = []
    collocation_historical = []
    y_dense_historical = []
    y_collocation_historical = []
    error_historical_dense = []
    error_historical_collocation = []

    y_true_dense = f(x_dense).detach().cpu()

    for i in tqdm(range(max_epochs)):

        y_true_collocation = f(x_collocation)
        y_estimated_collocation = model(x_collocation)

        loss = F.mse_loss(y_estimated_collocation, y_true_collocation)
        loss.backward()

        x_collocation.grad.data.mul_(-1)

        optimizer_min.step()
        optimizer_max.step()

        torch.clamp_(x_collocation.data, -1.0, 1.0)
        x_collocation.data.add_(torch.randn_like(x_collocation) * noise)

        optimizer_min.zero_grad()
        optimizer_max.zero_grad()

        # scheduler.step(loss)
        losses.append(loss.item())

        if animate_every is not None and (i % animate_every) == 0:
            y_dense = model(x_dense).detach().cpu()

            collocation_historical.append(x_collocation.detach().cpu())
            y_dense_historical.append(y_dense)
            error_historical_dense.append(
                F.mse_loss(y_dense, y_true_dense, reduction="none")
            )

            y_collocation = model(x_collocation)
            y_collocation_historical.append(y_collocation.detach().cpu())
            error_historical_collocation.append(
                F.mse_loss(y_collocation, y_true_collocation, reduction="none")
                .detach()
                .cpu()
            )

    # =============================== validation ============================

    y_true_collocation = f(x_collocation).detach().cpu()
    y_true_collocation_original = f(x_collocation_original).detach().cpu()
    y_estimated_collocation = model(x_collocation).detach().cpu()
    y_estimated_dense = model(x_dense).detach().cpu()

    if animate_every is not None:
        collocation_historical = torch.stack(collocation_historical).numpy()
        y_dense_historical = torch.stack(y_dense_historical).numpy()
        y_collocation_historical = torch.stack(y_collocation_historical).numpy()
        error_historical_dense = torch.stack(error_historical_dense).numpy()
        error_historical_collocation = torch.stack(error_historical_collocation).numpy()

    # ========================== send back to cpu ============================

    error_dense = (
        F.mse_loss(y_estimated_dense, y_true_dense, reduction="none")
        .detach()
        .cpu()
        .numpy()
    )
    error_collocation = (
        F.mse_loss(y_estimated_collocation, y_true_collocation, reduction="none")
        .detach()
        .cpu()
        .numpy()
    )
    y_estimated_collocation = y_estimated_collocation.detach().cpu().numpy()
    y_true_collocation = y_true_collocation.detach().cpu().numpy()
    y_true_collocation_original = y_true_collocation_original.detach().cpu().numpy()
    x_dense = x_dense.detach().cpu().numpy()
    x_collocation = x_collocation.detach().cpu().numpy()
    x_collocation_original = x_collocation_original.detach().cpu().numpy()

    y_estimated_dense = y_estimated_dense.detach().cpu().numpy()

    # ========================== plotting ============================
    if animate_every is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(x_dense, y_true_dense, label="true")
        ax1.plot(x_dense, y_estimated_dense, label="estimated")
        ax1.scatter(
            x_collocation_original,
            y_true_collocation_original,
            marker="o",
            label="collocation points (original)",
        )
        ax1.scatter(
            x_collocation,
            y_true_collocation,
            marker="x",
            label="collocation points (optimized)",
            c="r",
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.legend()

        ax2.plot(x_dense, error_dense)
        ax2.set_xlabel("x")
        ax2.set_ylabel("log error(x)")
        ax2.scatter(
            x_collocation,
            error_collocation,
            marker="x",
            label="collocation points (optimized)",
            c="r",
        )
        ax2.set_yscale("log")

        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_yscale("log")

        plt.show()

    # ========================== animations ===================

    if animate_every is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(x_dense, y_true_dense, label="true")
        (ln_est,) = ax1.plot([], [], label="estimated")
        s_est_colloc = ax1.scatter(
            [],
            [],
            marker="x",
            label="collocation points",
            c="r",
        )
        txt = ax1.text(
            0.1,
            0.1,
            "iteration x",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.legend()

        # error graph
        (ln_error,) = ax2.plot([], [], label="error")
        s_error_colloc = ax2.scatter(
            [],
            [],
            marker="x",
            label="collocation points",
            c="r",
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("error(x)")
        ax2.set_yscale("log")

        def init():

            ax2.set_xlim(-1, 1)
            ax2.set_ylim(error_historical_dense.min(), error_historical_dense.max())
            return (ln_est, ln_error, s_est_colloc, s_error_colloc, txt)

        def update(frame):
            i, (y, error_dense, x_colloc, y_colloc, error_colloc) = frame
            ln_est.set_data(x_dense, y)
            ln_error.set_data(x_dense, error_dense)
            s_est_colloc.set_offsets(np.hstack((x_colloc, y_colloc)))
            s_error_colloc.set_offsets(np.hstack((x_colloc, error_colloc)))

            txt.set_text(f"iteration: {i*animate_every}")
            return (ln_est, ln_error, s_est_colloc, s_error_colloc, txt)

        ani = FuncAnimation(
            fig,
            update,
            frames=enumerate(
                zip(
                    y_dense_historical,
                    error_historical_dense,
                    collocation_historical,
                    y_collocation_historical,
                    error_historical_collocation,
                )
            ),
            init_func=init,
            blit=True,
        )

        plt.show()
        writer = PillowWriter()

        makedirs("results", exist_ok=True)
        ani.save("results/active_collocation.gif", writer=writer)
