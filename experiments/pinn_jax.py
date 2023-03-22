import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd, jit, random, value_and_grad, vmap
from jax.config import config
from jax.example_libraries.optimizers import adam
from jax.nn import softplus
from tqdm import tqdm


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

    α = 1.0  # permeability
    μ = 1.0  # viscousity
    φ_inlet = 1.0  # flow velocity in x direction at inlet
    x_min = 0.0
    x_max = 1.0
    dx = 0.1
    y_min = 0.0
    y_max = 1.0
    dy = 0.1

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
    x_interior, y_interior = [
        x.reshape(-1)
        for x in jnp.meshgrid(
            jnp.arange(x_min, x_max + dx, dx), jnp.arange(y_min, y_max + dy, dy)
        )
    ]

    y_inlet = jnp.arange(y_min, y_max + dy, dy)
    y_outlet = y_inlet

    # loss function
    def loss_interior(params, x, y, α, μ):
        u, φ, γ, φx, φy, γx, γy = predict_func(params, x, y, α, μ)
        return (φx + γy) ** 2

    def loss_inlet(params, y, α, μ):
        u, φ, γ, φx, φy, γx, γy = predict_func(params, 0.0, y, α, μ)
        return (φ - φ_inlet) ** 2

    def loss_outlet(params, y, α, μ):
        u, φ, γ, φx, φy, γx, γy = predict_func(params, x_max, y, α, μ)
        return (u - 0) ** 2

    loss_interior_batched = vmap(
        loss_interior,
        in_axes=(None, 0, 0, None, None),
        out_axes=0,
    )
    loss_inlet_batched = vmap(
        loss_inlet,
        in_axes=(None, 0, None, None),
        out_axes=0,
    )
    loss_outlet_batched = vmap(
        loss_outlet,
        in_axes=(None, 0, None, None),
        out_axes=0,
    )

    def loss(params):
        l1 = loss_interior_batched(params, x_interior, y_interior, α, μ)
        l2 = loss_inlet_batched(params, y_inlet, α, μ)
        l3 = loss_outlet_batched(params, y_outlet, α, μ)

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

    fig, ax = plt.subplots()
    x, y = jnp.meshgrid(
        jnp.arange(x_min, x_max + dx, dx), jnp.arange(y_min, y_max + dy, dy)
    )
    α = 1.0
    μ = 1.0

    f = vmap(
        predict_func,
        in_axes=(None, 0, 0, None, None),
    )
    f = jit(vmap(f, in_axes=(None, 0, 0, None, None)))

    u, φ, γ, φx, φy, γx, γy = f(params, x, y, α, μ)

    x = np.array(x)
    y = np.array(y)
    φ = np.array(φ)
    γ = np.array(γ)

    im = ax.contourf(x, y, u)
    ax.streamplot(x, y, φ, γ, color="red")
    ax.scatter(x_interior, y_interior, label="interior")
    ax.scatter(jnp.zeros_like(y_inlet), y_inlet, label="inlet")
    ax.scatter(jnp.ones_like(y_inlet) * x_max, y_outlet, label="outlet")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.colorbar(im)

    plt.show()
