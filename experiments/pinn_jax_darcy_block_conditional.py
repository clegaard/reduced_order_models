from jax import jit, vmap, value_and_grad, jacfwd
from jax._src.numpy.lax_numpy import ones_like, reshape
import jax.numpy as jnp
from jax.nn import softplus
from jax import random
from jax import lax
from jax.example_libraries.optimizers import adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.config import config
import numpy as np


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
    config.update("jax_debug_nans", True)

    μ = 1.0  # viscosity
    φ_inlet = 1.0  # flow velocity in x direction at inlet
    u_outlet = 0.0
    inlet_block_idx = 3  # index(s) of block where liquid is injected
    # index(s) of blocks that are outlets
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
    A = A.at[1, 1].set(0.0)

    X, Y = jnp.meshgrid(
        jnp.arange(x_min, x_max + dx, dx), jnp.arange(y_min, y_max + dy, dy)
    )

    # indexing into blocks
    def get_block_idx(x, y):
        x_idx = jnp.floor_divide(x, block_width).astype(int)
        y_idx = jnp.floor_divide(y, block_width).astype(int)
        return x_idx + y_idx * n_blocks_x

    def permeablity(x, y):
        idx = get_block_idx(x, y)
        return A.reshape(-1)[idx]

    def is_inlet(x, y):
        idx = get_block_idx(x, y)
        c1 = lax.cond(idx == inlet_block_idx, lambda _: True, lambda _: False, None)
        c2 = lax.cond(x == 0.0, lambda _: True, lambda _: False, None)
        return jnp.logical_and(c1, c2)

    def is_outlet(x, x_end):
        return lax.cond(x == x_end, lambda _: True, lambda _: False, None)

    # training loop
    def loss(params, x, y):
        α = permeablity(x, y)

        u, φ, γ, φx, φy, γx, γy = predict_func(params, x, y, α, μ)

        def loss_inlet(φ):
            return (φ - φ_inlet) ** 2

        def loss_outlet(u):
            return (u - u_outlet) ** 2

        def loss_equations(vx, wy):
            return (vx + wy) ** 2

        l_inlet = lax.cond(is_inlet(x, y), lambda _: loss_inlet(φ), lambda _: 0.0, None)
        l_outlet = lax.cond(
            is_outlet(x, y), lambda _: loss_outlet(u), lambda _: 0.0, None
        )
        l_equations = loss_equations(φx, γy)

        return l_inlet + l_outlet + l_equations

    loss_batched = vmap(loss, (None, 0, 0), 0)
    loss_batched = vmap(loss_batched, (None, 1, 1), 1)

    @jit
    def update(step, params, opt_state):
        f = lambda params, x, y: jnp.sum(loss_batched(params, x, y))

        value, grads = value_and_grad(f)(params, X, Y)

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

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(
        np.arange(x_min, x_max + dx * 0.01, dx * 0.01),
        np.arange(y_min, y_max + dy * 0.01, dy * 0.01),
    )
    α = vmap(vmap(permeablity, (0, 0), 0), (1, 1), 1)(X, Y)

    f = vmap(predict_func, (None, 0, 0, 0, None))
    f = vmap(f, (None, 1, 1, 1, None), 1)

    u, φ, γ, φx, φy, γx, γy = f(params, X, Y, α, μ)

    # X = (
    #     X.reshape(n_blocks_x, n_blocks_y, X.shape[1], X.shape[2])
    #     .transpose(0, 2, 1, 3)
    #     .reshape(n_blocks_x * X.shape[1], n_blocks_y * X.shape[2])
    # )
    # Y = (
    #     Y.reshape(n_blocks_x, n_blocks_y, Y.shape[1], Y.shape[2])
    #     .transpose(0, 2, 1, 3)
    #     .reshape(n_blocks_x * Y.shape[1], n_blocks_y * Y.shape[2])
    # )
    # u = (
    #     u.reshape(n_blocks_x, n_blocks_y, u.shape[1], u.shape[2])
    #     .transpose(0, 2, 1, 3)
    #     .reshape(n_blocks_x * u.shape[1], n_blocks_y * u.shape[2])
    # )

    im = ax.contourf(X, Y, u, levels=100)
    ax.streamplot(X, Y, φ, γ, color="red")
    # ax.scatter(X, Y, label="interior")
    # # ax.scatter(jnp.zeros_like(y_inlets), y_inlets, label="inlet")
    # # ax.scatter(jnp.ones_like(y_inlets) * x_max, y_outlets, label="outlet")
    # ax.legend()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(im)

    plt.show()
