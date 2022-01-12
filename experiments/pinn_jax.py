from jax import jit, vmap, value_and_grad, jacfwd
from jax._src.numpy.lax_numpy import reshape
import jax.numpy as jnp
from jax.nn import softplus
from jax import random
from jax.experimental.optimizers import adam
from tqdm import tqdm
import matplotlib.pyplot as plt


def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict_func(parameters, x, y, α, μ):

    # forward pass through network
    def f(parameters, x, y, α, μ):
        activations = jnp.concatenate((x, y, α, μ))

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
    # possible use of https://github.com/kach/jax.value_and_jacfwd/blob/main/value_and_jacfwd.py
    j = jacfwd(h, argnums=[1, 2])

    (ux, uy), (φx, φy), (γx, γy) = j(parameters, x, y, α, μ)
    u = f(parameters, x, y, α, μ)

    return u, φx, φy, γx, γy


if __name__ == "__main__":

    α = 1.0  # permeability
    μ = 1.0  # viscousity
    x_min = 0.0
    x_max = 1.0
    dx = 0.1
    y_min = 0.0
    y_max = 1.0
    dy = 0.1

    n_training_steps = 1
    learning_rate = 1e-4

    key = random.PRNGKey(1)

    # construct network
    layer_sizes = [4, 32, 1]
    params = initialize_mlp(layer_sizes, key)

    # training
    opt_init, opt_update, get_params = adam(learning_rate)
    opt_state = opt_init(params)

    # training data

    x_interior, y_interior = jnp.meshgrid(
        jnp.arange(x_min, x_max, dx), jnp.arange(y_min, y_max, dy)
    )
    x_interior = x_interior.reshape(-1, 1)
    y_interior = y_interior.reshape(-1, 1)
    α_train = jnp.ones_like(x_interior) * α
    μ_train = jnp.ones_like(x_interior) * μ

    # loss function
    def loss_interior(params, x, y, α, μ):
        u, φx, φy, γx, γy = predict_func(params, x, y, α, μ)
        return jnp.linalg.norm(u - 1.0)

    loss_interior_batched = vmap(
        loss_interior,
        in_axes=(None, 0, 0, 0, 0),
        out_axes=0,
    )

    def loss(params):
        value = loss_interior_batched(params, x_interior, y_interior, α_train, μ_train)
        return jnp.sum(value)

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
    x, y = jnp.meshgrid(jnp.arange(x_min, x_max, dx), jnp.arange(y_min, y_max, dy))
    α = 1.0
    μ = 1.0

    f = vmap(
        predict_func,
        in_axes=(None, 0, 0, None, None),
    )
    f = vmap(f, in_axes=(None, 0, 0, None, None))

    u = f(params, x, y, α, μ)

    plt.show()
