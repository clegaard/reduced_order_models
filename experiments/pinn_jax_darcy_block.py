import numpy as np
from jax import jit, vmap, value_and_grad, jacfwd
import jax.numpy as jnp
from jax.nn import softplus
from jax import random
from jax.experimental.optimizers import adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.config import config


def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict_func(parameters, x, y, α, μ):

    # forward pass through network
    def f(parameters, x, y, α, μ):
        # activations = jnp.concatenate((x, y, α, μ))
        activations = jnp.array((x, y, α, μ))

        for w, b in parameters[:-1]:
            activations = softplus(jnp.dot(w, activations) + b)

        w, b = parameters[-1]
        u = jnp.sum(jnp.dot(w, activations) + b)
        return u

    # spatial derivatives of pressure
    g = value_and_grad(f, argnums=[1, 2])  # returns u, (ux, uy)

    # velocity using darcys law
    def h(parameters, x, y, α, μ):
        u, (ux, uy) = g(parameters, x, y, α, μ)
        φ = -α / μ * ux  # velocity in x direction
        γ = -α / μ * uy  # velocity in y direction
        return u, φ, γ

    # derivative of velocity
    j = jacfwd(h, argnums=[1, 2])

    (ux, uy), (φx, φy), (γx, γy) = j(parameters, x, y, α, μ)
    # can possible be optimized by using https://github.com/kach/jax.value_and_jacfwd/blob/main/value_and_jacfwd.py
    u, φ, γ = h(parameters, x, y, α, μ)

    return u, φ, γ, φx, φy, γx, γy


if __name__ == "__main__":

    config.update("jax_debug_nans", False)

    μ = 1.0  # viscosity
    φ_inlet = 1.0  # flow velocity in x direction at inlet
    inlet_block_idx = jnp.asarray([3])  # index(s) of block where liquid is injected
    outlet_block_idx = jnp.asarray([2, 5, 8])  # index(s) of blocks that are outlets
    dx = 0.1
    dy = 0.1
    n_blocks_x = 3
    n_blocks_y = 3
    block_width = 1 / 3
    x_min = 0.0  # start of domain in x direction
    x_max = x_min + n_blocks_x * block_width  # end of domain in x direction
    y_min = 0.0  # start of domain in y direction
    y_max = y_min + n_blocks_y * block_width  # end of domain in y direction

    n_training_steps = 100000
    learning_rate = 1e-4

    key = random.PRNGKey(1)

    # construct network
    layer_sizes = [4, 32, 1]
    params = initialize_mlp(layer_sizes, key)

    # training
    opt_init, opt_update, get_params = adam(learning_rate)
    opt_state = opt_init(params)

    # training data
    A = jnp.ones((3, 3))
    A = A.at[1, 1].set(0.005)

    X, Y = jnp.meshgrid(
        jnp.arange(x_min, x_max + dx, dx), jnp.arange(y_min, y_max + dy, dy)
    )

    def permeablity(x, y):
        x_idx = jnp.floor_divide(x, block_width).astype(int)
        y_idx = jnp.floor_divide(y, block_width).astype(int)
        return A[x_idx, y_idx]

    α = vmap(vmap(permeablity, (0, 0), 0), (1, 1), 1)(X, Y)

    y_inlets = jnp.arange(block_width, 2 * block_width + dy, dy)
    α_inlets = vmap(permeablity, (0, 0), 0)(jnp.zeros_like(y_inlets), y_inlets)

    y_outlets = jnp.arange(y_min, y_max + dy, dy)
    α_outlets = vmap(permeablity, (0, 0), 0)(jnp.zeros_like(y_outlets), y_outlets)

    # loss function
    def loss_interior(params, x, y, α, μ):
        _, _, _, φx, _, _, γy = predict_func(params, x, y, α, μ)
        return (φx + γy) ** 2

    def loss_inlet(params, y, α, μ):
        _, φ, _, _, _, _, _ = predict_func(params, 0.0, y, α, μ)
        return (φ - φ_inlet) ** 2

    def loss_outlet(params, y, α, μ):
        u, _, _, _, _, _, _ = predict_func(params, x_max, y, α, μ)
        return (u - 0) ** 2

    loss_interior_batched = vmap(loss_interior, (None, 0, 0, 0, None), 0)
    loss_interior_batched = vmap(loss_interior_batched, (None, 1, 1, 1, None), 1)

    loss_inlet_batched = vmap(loss_inlet, (None, 0, 0, None), 0)

    loss_outlet_batched = vmap(loss_outlet, (None, 0, 0, None), 0)

    def loss(params):

        l1 = loss_interior_batched(params, X, Y, α, μ)

        l2 = loss_inlet_batched(params, y_inlets, α_inlets, μ)

        l3 = loss_outlet_batched(params, y_outlets, α_outlets, μ)

        return jnp.sum(l1) + jnp.sum(l2) + jnp.sum(l3)

    # training loop
    @jit
    def update(step, params, opt_state):
        value, grads = value_and_grad(loss)(params)
        opt_state = opt_update(step, grads, opt_state)
        params = get_params(opt_state)
        return params, opt_state, value

    losses = []

    for step in tqdm(range(n_training_steps), desc="training iteration"):

        params, opt_state, value = update(step, params, opt_state)

        losses.append(value)

    # ================== plotting ====================
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_yscale("log")

    X, Y = np.meshgrid(
        np.arange(x_min, x_max + 0.1, dx * 0.1),
        np.arange(y_min, y_max + dy * 0.1, dy * 0.1),
    )
    α = vmap(vmap(permeablity, (0, 0), 0), (1, 1), 1)(X, Y)

    f = vmap(predict_func, (None, 0, 0, 0, None))
    f = vmap(f, (None, 1, 1, 1, None), 1)
    u, φ, γ, φx, φy, γx, γy = f(params, X, Y, α, μ)

    # pressure map
    # fig, ax = plt.subplots()
    # speed = speed = np.sqrt(φ ** 2 + γ ** 2)
    # lw = 1 * speed / speed.max()
    # im = ax.contourf(X, Y, u, levels=100)
    # ax.streamplot(X, Y, φ, γ, color="red", linewidth=lw)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # plt.colorbar(im)

    # pressure map (quiver)
    fig, ax = plt.subplots()
    im = ax.contourf(X, Y, u, levels=100)
    ax.quiver(X, Y, φ, γ, color="red", scale_units="xy", angles="xy")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(im)

    # velocity map
    fig, ax = plt.subplots()

    speed = np.sqrt(φ ** 2 + γ ** 2)

    im = ax.contourf(X, Y, speed, levels=100)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(im)

    plt.show()
