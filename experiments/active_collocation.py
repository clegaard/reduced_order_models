from symengine import var, diff, lambdify
from sympy import plot, simplify
import torch
from torch import optim
from torch._C import ScriptFunction
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.animation import FuncAnimation


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
    max_epochs = 2000
    animate_every = int(max_epochs / 100)

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
    error_historical = []

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
            error_historical.append(F.mse_loss(y_dense, y_true_dense, reduction="none"))

    # for _ in tqdm(range(max_epochs)):

    #     y_true_collocation = f(x_collocation)
    #     y_estimated_collocation = model(x_collocation)

    #     loss = F.mse_loss(y_estimated_collocation, y_true_collocation)
    #     loss.backward()

    #     x_collocation.grad.data.mul_(-1)

    #     optimizer_max.step()

    #     torch.clamp_(x_collocation.data, -1.0, 1.0)
    #     # x_collocation.data.add_(torch.randn_like(x_collocation) * noise)

    #     optimizer_max.zero_grad()

    # =============================== validation ============================

    y_true_collocation = f(x_collocation)
    y_true_collocation_original = f(x_collocation_original)
    y_estimated_collocation = model(x_collocation)
    y_estimated_dense = model(x_dense)

    if animate_every is not None:
        collocation_historical = torch.stack(collocation_historical).numpy()
        y_dense_historical = torch.stack(y_dense_historical).numpy()
        error_historical = torch.stack(error_historical).numpy()

    # ========================== send back to cpu ============================

    y_estimated_collocation = y_estimated_collocation.detach().cpu()
    y_estimated_dense = y_estimated_dense.detach().cpu()
    y_true_collocation = y_true_collocation.detach().cpu()
    y_true_collocation_original = y_true_collocation_original.detach().cpu()
    x_dense = x_dense.detach().cpu()
    x_collocation = x_collocation.detach().cpu()
    x_collocation_original = x_collocation_original.detach().cpu()
    error_dense = F.mse_loss(y_estimated_dense, y_true_dense, reduction="none")
    error_collocation = F.mse_loss(
        y_estimated_collocation, y_true_collocation, reduction="none"
    )

    # ========================== plotting ============================

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
        fig, ax = plt.subplots(sharex=True)
        ax.plot(x_dense, y_true_dense, label="true")
        # (ln,) = ax.plot(x_dense, y_estimated_dense, label="estimated")
        (ln,) = ax.plot([], [], label="estimated")
        txt = ax.text(
            0.75,
            0.75,
            "iteration x",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        # def init():
        #     ax.set_xlim(-1.0, 1.0)
        #     return ln

        def update(frame):
            i, y = frame
            ln.set_data(x_dense, y)
            txt.set_text(f"iteration: {i*animate_every}")
            return (ln, txt)

        ani = FuncAnimation(
            fig,
            update,
            frames=enumerate(y_dense_historical),
            # init_func=init,
            blit=True,
        )

        plt.show()
        # ax.scatter(
        #     x_collocation_original,
        #     y_true_collocation_original,
        #     marker="o",
        #     label="collocation points (original)",
        # )
        # ax.scatter(
        #     x_collocation,
        #     y_true_collocation,
        #     marker="x",
        #     label="collocation points (optimized)",
        #     c="r",
        # )
        # ax.set_xlabel("x")
        # ax.set_ylabel("f(x)")
        # ax.legend()

        # ax2.plot(x_dense, error_dense)
        # ax2.set_xlabel("x")
        # ax2.set_ylabel("log error(x)")
        # ax2.scatter(
        #     x_collocation,
        #     error_collocation,
        #     marker="x",
        #     label="collocation points (optimized)",
        #     c="r",
        # )
        # ax2.set_yscale("log")
