"""Eq. (1): remask a sequence on an edit set S."""

from __future__ import annotations

import jax.numpy as jnp


def as_bool_mask(S, L: int):
    """Bool mask of shape (..., L). Accepts a bitmask or a length-L 0/1 vector."""
    S = jnp.asarray(S)
    if S.ndim >= 1 and S.shape[-1] == L:
        return S.astype(bool)
    bits = (jnp.asarray(S, dtype=jnp.int32)[..., None] >> jnp.arange(L, dtype=jnp.int32)) & 1
    return bits.astype(bool)


def as_bits(S, L: int):
    """Integer bitmask: bit i set iff position i is in S (0-indexed)."""
    S = jnp.asarray(S)
    if S.ndim >= 1 and S.shape[-1] == L:
        powers = (1 << jnp.arange(L, dtype=jnp.int32)).astype(jnp.int32)
        return (S.astype(jnp.int32) * powers).sum(axis=-1).astype(jnp.int32)
    return S.astype(jnp.int32)


def remask(x, S, mask_id):
    """Eq. (1): x̃_i = MASK if i in S else x_i. Positions are 0-indexed."""
    x = jnp.asarray(x)
    L = x.shape[-1]
    mask = as_bool_mask(S, L)
    return jnp.where(mask, jnp.asarray(mask_id, dtype=x.dtype), x)
