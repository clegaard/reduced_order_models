from typing_extensions import Required
from numpy import dtype
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n_train_iterations = 10000
    n_train_steps = 20
    device = "cuda"

    Δt = 0.01
    u, t, x, M = heateq_1d_square_implicit_euler_matrix(
        t_start=0.0,
        t_end=1.49,
        Δt=Δt,
        x_start=0.0,
        x_end=1.0,
        Δx=0.01,
        α=-0.5,
    )

    # TODO
    u = u.T
    # end TODO

    u = torch.tensor(u, dtype=torch.float32).to(device)
    u = u[:, : n_train_steps + 1]
    t = t[: n_train_steps + 1]
    u0 = u[:, 0].to(device)

    # M = torch.randn(M.shape, requires_grad=True).to(device)
    M = torch.nn.Linear(M.shape[0], M.shape[0]).to(device)
    loss = None
    losses = []
    opt = torch.optim.Adam(M.parameters())

    # ---------------------- foward euler explicit, full system ----------------------------------
    u_estimated = None

    for _ in tqdm(range(n_train_iterations), desc="training iteration"):
        cur = u0
        next = None
        uu = [u0]

        for _ in range(n_train_steps):
            next = cur + M(cur) * Δt
            uu.append(next)
            cur = next

        u_estimated = torch.stack(uu, axis=1)
        loss = F.mse_loss(u_estimated, u)
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()

    # u_step = u.unfold(dimension=1, size=1, step=1)
    # for _ in tqdm(range(n_train_iterations), desc="training iteration"):
    #     next = u_step + M(u_step) * Δt
    #     u_estimated_step = u_estimated.unfold(dimension=1, size=1, step=1)

    #     loss = F.mse_loss(u_estimated, u)
    #     loss.backward()
    #     losses.append(loss.item())
    #     opt.step()
    #     opt.zero_grad()

    # ================================ plotting ================================

    u_estimated = u_estimated.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")

    heatmap_1d(u_estimated.T, x, t)
    plt.show()
