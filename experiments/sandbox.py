import jax
from jax import vmap
import jax.numpy as jnp


def f(x, y, a):
    b = 10


y = jnp.ones((10, 10, 9))
x = jnp.ones((10, 10, 9))
a = jnp.ones(9)

# f(x, y)   # function that takes (), () , ()
g = vmap(f, in_axes=(0, 0, None))  # function that takes (9) (9), (9)
h = vmap(g, in_axes=(0, 0, None))  # function that takes (9) (9), (9)
k = vmap(h, in_axes=(2, 2, 0))  # function that takes (9) (9), (9)
k(x, y, a)


y = jnp.ones((9, 10, 10))
x = jnp.ones((9, 10, 10))
a = jnp.ones(9)

# f(x, y)   # function that takes (), () , ()
g = vmap(f, in_axes=(0, 0, 0))  # function that takes (9) (9), (9)
h = vmap(g, in_axes=(1, 1, None))  # function that takes (9) (9), (9)
k = vmap(h, in_axes=(2, 2, None))  # function that takes (9) (9), (9)
k(x, y, a)
