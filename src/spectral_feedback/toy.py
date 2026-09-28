"""Sparse-interaction sequence reward and a planted Fourier-sparse set function."""

from __future__ import annotations

import jax.numpy as jnp

from spectral_feedback.fourier import fourier_inverse


# Empty set, two singletons, one pair: 4 known coefficients on L=8.
PLANTED_T = (0, 1 << 0, 1 << 1, (1 << 2) | (1 << 3))
PLANTED_F = (1.0, 0.5, -0.4, 0.8)


def planted_coefficients(L: int = 8):
    """Dense F table with 4 nonzero Walsh-Hadamard coefficients."""
    F = jnp.zeros((2**L,), dtype=jnp.float64)
    T = jnp.asarray(PLANTED_T, dtype=jnp.int32)
    vals = jnp.asarray(PLANTED_F, dtype=jnp.float64)
    return F.at[T].set(vals)


def planted_set_function(L: int = 8):
    """Direct set function f_true(S) with known sparse Fourier support (eq. 3 inverse)."""
    F = planted_coefficients(L)
    return fourier_inverse(F), F


def sequence_reward(x, target, weights, pairs, pair_coefs):
    """r(x) = sum_i w_i 1[x_i = t_i] + sum_{(i,j) in P} c_ij 1[x_i = x_j]."""
    x = jnp.asarray(x)
    target = jnp.asarray(target)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    pairs = jnp.asarray(pairs, dtype=jnp.int32)
    pair_coefs = jnp.asarray(pair_coefs, dtype=jnp.float64)
    match = jnp.sum(weights * (x == target).astype(jnp.float64))
    i = pairs[:, 0]
    j = pairs[:, 1]
    pair = jnp.sum(pair_coefs * (x[i] == x[j]).astype(jnp.float64))
    return match + pair


def default_toy(L: int = 8, vocab_size: int = 4):
    """Length-L toy: hidden target, 3 pair interactions, MASK is the last token."""
    target = jnp.arange(L, dtype=jnp.int32) % (vocab_size - 1)
    weights = jnp.ones((L,), dtype=jnp.float64)
    pairs = jnp.asarray([[0, 1], [2, 3], [4, 6]], dtype=jnp.int32)
    pair_coefs = jnp.asarray([1.5, 1.5, 1.0], dtype=jnp.float64)
    mask_id = vocab_size - 1

    def reward_fn(x):
        return sequence_reward(x, target, weights, pairs, pair_coefs)

    return {
        "L": L,
        "vocab_size": vocab_size,
        "mask_id": mask_id,
        "target": target,
        "weights": weights,
        "pairs": pairs,
        "pair_coefs": pair_coefs,
        "reward_fn": reward_fn,
    }
