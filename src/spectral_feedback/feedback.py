"""Algorithm 1 (Spectral Edit-Set Selection) and Algorithm 2 (Protein Model Feedback)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectral_feedback.mask import remask
from spectral_feedback.recover import argmax_edit_set, fit_truncated_fourier, sample_edit_sets


def spectral_edit_set_selection(
    key,
    L: int,
    D: int,
    gamma: float,
    d: int,
    s: int,
    k: int,
    value_fn=None,
    S_bits=None,
    values=None,
):
    """Algorithm 1.

    Sample D Bernoulli(gamma) edit-sets, evaluate r̃_j = f(S_j, x) (eq. 2),
    recover a truncated Fourier model, return S* = argmax_{|S|<=k} fhat(S).
    """
    if S_bits is None:
        key, k_s = jax.random.split(key)
        S_bits, _mask = sample_edit_sets(k_s, L, D, gamma)
    else:
        S_bits = jnp.asarray(S_bits, dtype=jnp.int32)
    if values is None:
        if value_fn is None:
            raise ValueError("value_fn is required when values are not provided")
        values = jax.vmap(value_fn)(S_bits)
    else:
        values = jnp.asarray(values, dtype=jnp.float64)
    _T, _F, F_dense = fit_truncated_fourier(S_bits, values, L, d, s)
    S_star = argmax_edit_set(F_dense, k, L=L)
    return S_star, F_dense


def protein_model_feedback(
    key,
    N: int,
    L: int,
    mask_id: int,
    sample_fn,
    edit_select_fn,
):
    """Algorithm 2.

    Start fully masked with S = [L]. For i = 1..N-1: sample x ~ p(· | S, x),
    then S = EditSelection(x). Final sample x ~ p(· | S, x).
    """
    x = jnp.full((L,), mask_id, dtype=jnp.int32)
    S_bits = jnp.int32((1 << L) - 1)
    for _ in range(N - 1):
        key, k_samp, k_edit = jax.random.split(key, 3)
        xtilde = remask(x, S_bits, mask_id)
        x = sample_fn(xtilde, k_samp)
        S_bits = edit_select_fn(x, k_edit)
        S_bits = jnp.asarray(S_bits, dtype=jnp.int32)
    key, k_samp = jax.random.split(key)
    xtilde = remask(x, S_bits, mask_id)
    x = sample_fn(xtilde, k_samp)
    return x
