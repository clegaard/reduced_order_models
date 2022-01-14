from jax import jit, vmap, value_and_grad, jacfwd
from jax._src.numpy.lax_numpy import ones_like, reshape
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

    n_training_steps = 1000
    learning_rate = 1e-4

    key = random.PRNGKey(1)

    # construct network
    layer_sizes = [4, 32, 1]
    params = initialize_mlp(layer_sizes, key)

    # training
    opt_init, opt_update, get_params = adam(learning_rate)
    opt_state = opt_init(params)

    # training data
    X = []
    Y = []
    A = []
    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            x, y = jnp.meshgrid(
                jnp.arange(i * block_width, i * block_width + block_width, dx),
                jnp.arange(j * block_width, j * block_width + block_width, dy),
            )
            # A.append(n_blocks_x * n_blocks_y + 1)
            A.append(1.0)
            X.append(x)
            Y.append(y)
    X = jnp.stack(X)
    Y = jnp.stack(Y)
    A = jnp.stack(A)

    α_inlets = A[inlet_block_idx]
    y_inlets = Y[inlet_block_idx, 0]

    α_outlets = A[outlet_block_idx]
    y_outlets = Y[outlet_block_idx, -1]

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

    # loss_interior_batched = vmap(
    #     loss_interior,
    #     (None, 0, 0, 0, None),
    # )
    # loss_interior_batched = vmap(
    #     loss_interior_batched,
    #     (None, 1, 1, None, None),
    # )
    # loss_interior_batched = vmap(
    #     loss_interior_batched,
    #     (None, 2, 2, None, None),
    # )

    loss_interior_batched = vmap(loss_interior, (None, 0, 0, 0, None), 0)
    loss_interior_batched = vmap(loss_interior_batched, (None, 1, 1, None, None), 1)
    loss_interior_batched = vmap(loss_interior_batched, (None, 2, 2, None, None), 2)

    loss_inlet_batched = vmap(loss_inlet, (None, 0, 0, None), 0)
    loss_inlet_batched = vmap(loss_inlet_batched, (None, 1, None, None), 1)

    loss_outlet_batched = vmap(loss_outlet, (None, 0, 0, None), 0)
    loss_outlet_batched = vmap(loss_outlet_batched, (None, 1, None, None), 1)

    def loss(params):

        losses = []

        l1 = loss_interior_batched(params, X, Y, A, μ)

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

    fig, ax = plt.subplots()
    μ = 1.0

    f = vmap(predict_func, (None, 0, 0, 0, None))
    f = vmap(f, (None, 1, 1, None, None), 1)
    f = vmap(f, (None, 2, 2, None, None), 2)

    u, φ, γ, φx, φy, γx, γy = f(params, X, Y, A, μ)

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

    im = ax.contourf(X[0], Y[0], u[0])
    im = ax.contourf(X[1], Y[1], u[1])
    # ax.streamplot(x, y, φ, γ, color="red")
    ax.scatter(X, Y, label="interior")
    # ax.scatter(jnp.zeros_like(y_inlets), y_inlets, label="inlet")
    # ax.scatter(jnp.ones_like(y_inlets) * x_max, y_outlets, label="outlet")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # plt.colorbar(im)

    plt.show()
