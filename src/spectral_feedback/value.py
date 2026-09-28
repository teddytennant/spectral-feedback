"""Eq. (2): Monte Carlo edit-set value functions favg and fmax."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectral_feedback.mask import remask


def _rewards(x, S, key, n: int, sample_fn, reward_fn, mask_id: int):
    xtilde = remask(x, S, mask_id)
    keys = jax.random.split(key, n)
    zs = jax.vmap(lambda k: sample_fn(xtilde, k))(keys)
    return jax.vmap(reward_fn)(zs)


def favg(x, S, key, n: int, sample_fn, reward_fn, mask_id: int):
    """Eq. (2): favg(S, x) = (1/n) sum_{i=1}^n r(z_i), z_i ~ p(· | remask(x, S))."""
    return jnp.mean(_rewards(x, S, key, n, sample_fn, reward_fn, mask_id))


def fmax(x, S, key, n: int, sample_fn, reward_fn, mask_id: int):
    """Eq. (2): fmax(S, x) = max{r(z_1), ..., r(z_n)}, z_i ~ p(· | remask(x, S))."""
    return jnp.max(_rewards(x, S, key, n, sample_fn, reward_fn, mask_id))
