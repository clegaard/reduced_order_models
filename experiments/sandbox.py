from jax import make_jaxpr, jit, grad, vmap, value_and_grad
import jax.numpy as jnp
from jax.nn import softplus


from jax import random

if __name__ == "__main__":
    key = random.PRNGKey(1)

    def initialize_mlp(sizes, key):
        keys = random.split(key, len(sizes))

        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(
                b_key, (n,)
            )

        return [
            initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def predict_fun(parameters, x, y, a):
        activations = jnp.concatenate((x, y, a))

        for w, b in parameters[:-1]:
            activations = softplus(jnp.dot(w, activations) + b)

        w, b = parameters[-1]

        return jnp.sum(jnp.dot(w, activations) + b)

    layer_sizes = [3, 128, 1]

    params = initialize_mlp(layer_sizes, key)

    predict_batched = vmap(predict_fun, in_axes=(None, 0, 0, 0), out_axes=0)

    g = vmap(
        value_and_grad(predict_fun, argnums=[1, 2]), in_axes=(None, 0, 0, 0), out_axes=0
    )

    value, (dx, dy) = g(params, jnp.ones((3, 1)), jnp.ones((3, 1)), jnp.ones((3, 1)))
