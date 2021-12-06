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


def f(x, e=3):
    return x ** e * (x ** 2 - 1)


# def g(x):
#     return x ** 9 * (156 * x ** 2 - 110)


# def f(x):
#     return torch.sin(x * 31)


if __name__ == "__main__":

    hidden_dim = 128

    dx_dense = 0.001
    dx_collocation = 0.01
    device = "cuda"
    max_epochs = 1000

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

    # ================== optimizers ==================
    optimizer_min = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_max = torch.optim.Adam((x_collocation,), lr=0.0)

    # scheduler = ReduceLROnPlateau(optimizer_min, "min", patience=100)

    losses = []

    for _ in tqdm(range(max_epochs)):

        y_true_collocation = f(x_collocation)
        y_estimated_collocation = model(x_collocation)

        loss = F.mse_loss(y_estimated_collocation, y_true_collocation)
        loss.backward()

        x_collocation.grad.data.mul_(-1)

        optimizer_min.step()
        optimizer_max.step()

        torch.clamp_(x_collocation.data, -1.0, 1.0)
        # x_collocation.requires_grad = True

        optimizer_min.zero_grad()
        optimizer_max.zero_grad()

        # scheduler.step(loss)
        losses.append(loss.item())

    # =============================== validation ============================
    y_true_dense = f(x_dense)
    y_estimated_dense = model(x_dense)
    y_true_collocation = f(x_collocation)
    y_true_collocation_original = f(x_collocation_original)
    y_estimated_collocation = model(x_collocation)

    # ========================== send back to cpu ============================

    y_estimated_collocation = y_estimated_collocation.detach().cpu()
    y_estimated_dense = y_estimated_dense.detach().cpu()
    y_true_dense = y_true_dense.detach().cpu()
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
    ax2.set_ylabel("error(x)")
    ax2.scatter(
        x_collocation,
        error_collocation,
        marker="x",
        label="collocation points (optimized)",
        c="r",
    )

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale("log")

    plt.show()
