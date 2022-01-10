from jax import make_jaxpr, jit, grad, vmap, value_and_grad, jacfwd
from jax.core import AvalMapHandlerPair
import jax.numpy as jnp
from jax.nn import softplus
from jax import random
import matplotlib.pyplot as plt
import functools
import jax


def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict_pressure(parameters, x, y, α, μ):
    activations = jnp.concatenate((x, y, α, μ))

    for w, b in parameters[:-1]:
        activations = softplus(jnp.dot(w, activations) + b)

    w, b = parameters[-1]

    return jnp.sum(jnp.dot(w, activations) + b)


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
    y_min = 0.0
    y_max = 1.0

    key = random.PRNGKey(1)

    layer_sizes = [4, 128, 1]

    params = initialize_mlp(layer_sizes, key)

    # pressures_and_spatial_derivatives = vmap(
    #     value_and_grad(predict_pressure, argnums=[1, 2]),
    #     in_axes=(None, 0, 0, 0, 0),
    #     out_axes=0,
    # )

    pressures_and_spatial_derivatives = jit(
        vmap(predict_func, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    )
    #     out_axes=0,))

    # training
    u, φx, φy, γx, γy = pressures_and_spatial_derivatives(
        params,
        jnp.ones((3, 1)) * 0.0,
        jnp.ones((3, 1)) * 0.0,
        jnp.ones((3, 1)) * 1.0,
        jnp.ones((3, 1)) * 1.0,
    )

    # plotting

    # xx, yy = jnp.meshgrid(
    #     jnp.arange(x_min, x_max, 0.01), jnp.arange(y_min, y_max, 0.01)
    # )
    # x = xx.reshape(-1, 1)
    # y = yy.reshape(-1, 1)

    # u, _ = pressures_and_spatial_derivatives(
    #     params, x, y, jnp.ones_like(x) * α, jnp.ones_like(y) * μ
    # )
    # uu = u.reshape(100, 100)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(xx, yy, uu, linewidth=0, antialiased=False)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("p(x,y)")

    # plt.show()
