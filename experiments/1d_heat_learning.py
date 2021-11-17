from typing_extensions import Required
from numpy import dtype
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import heateq_1d_square_implicit_euler_matrix
from plotting import heatmap_1d
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_

if __name__ == "__main__":

    n_train_iterations = 1000
    n_train_steps = 149
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

    assert n_train_steps <= len(
        t
    ), f"training data must have a length greater or equal than the number of training steps, actual is {len(t)} and {n_train_steps}"

    # TODO
    u = u.T
    # end TODO

    u = torch.tensor(u, dtype=torch.float32).to(device)
    u = u[:, : n_train_steps + 1]
    t = t[: n_train_steps + 1]
    u0 = u[:, 0].to(device)

    M = torch.nn.Linear(M.shape[0], M.shape[0], bias=False).to(device)
    xavier_uniform_(M.weight)
    loss = None
    losses = []
    opt = torch.optim.Adam(M.parameters())

    # ---------------------- foward euler explicit, full system ----------------------------------

    for _ in tqdm(range(n_train_iterations), desc="training iteration"):
        # map each snapshot x_k to an estimate of the next state x_k+1
        # operation can be carried out in parallel for each snapshot
        u_next = u + M(u.T).T * Δt
        # u = [x0 x1 ... xn] , u_next = [x1 x2 ... xn+1]
        loss = F.mse_loss(u_next[:, :-1], u[:, 1:])
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()

    # ================================ validation ==============================
    u_estimated_step = torch.concat((u[:, :1], u_next[:, :-1]), dim=1)

    cur = u[:, :1]
    next = None
    us = [cur]
    for _ in tqdm(t):
        next = cur + M(cur.T).T * Δt
        us.append(next)
        cur = next
    u_estimated_sim = torch.concat(us, dim=1)
    assert u_estimated_step.shape == u_estimated_sim.shape

    # ================================ plotting ================================

    u = u.detach().cpu().numpy()
    u_estimated_step = u_estimated_step.detach().cpu().numpy()
    u_estimated_sim = u_estimated_sim.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss, single step")

    fig, ax = heatmap_1d(u.T, x, t)
    ax.set_title("true")

    fig, ax = heatmap_1d(u_estimated_step.T, x, t)
    ax.set_title("predicted (step)")

    fig, ax = heatmap_1d(u_estimated_sim.T, x, t)
    ax.set_title("predicted (sim)")
    plt.show()
