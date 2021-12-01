import jax.numpy as jnp
from jax import value_and_grad, jit
from jax.random import normal
from jax.lax import scan
from jax.flatten_util import ravel_pytree


def NIGupdate(bel, phi, reward):
    mu, Sigma, a, b = bel
    Lambda = jnp.linalg.inv(Sigma)
    Lambda_update = jnp.outer(phi, phi) + Lambda
    Sigma_update = jnp.linalg.inv(Lambda_update)
    mu_update = Sigma_update @ (Lambda @ mu + phi * reward)
    a_update = a + 1 / 2
    b_update = b + (reward ** 2 + mu.T @ Lambda @ mu - mu_update.T @ Lambda_update @ mu_update) / 2
    bel = (mu_update, Sigma_update, a_update, b_update)
    return bel


def convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full):
    params = jnp.matmul(params_subspace, projection_matrix) + params_full
    return params


def generate_random_basis(key, d, D):
    projection_matrix = normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


def train(state, loss_fn, nepochs=300, has_aux=True):
    @jit
    def step(state, _):
        grad_fn = value_and_grad(loss_fn, has_aux=has_aux)
        val, grads = grad_fn(state.params)
        loss = val[0] if has_aux else val
        state = state.apply_gradients(grads=grads)
        flat_params, _ = ravel_pytree(state.params)
        return state, {"loss": loss, "params": flat_params}

    state, metrics = scan(step, state, jnp.empty(nepochs))

    return state, metrics
