"""Independent-token absorbing-mask reverse kernel (not a protein model)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def forward_mask(x, key, p: float, mask_id: int):
    """Forward: each token independently becomes MASK with probability p."""
    x = jnp.asarray(x)
    drop = jax.random.bernoulli(key, p, x.shape)
    return jnp.where(drop, jnp.asarray(mask_id, dtype=x.dtype), x)


def sample_reverse(xtilde, key, target, vocab_size: int, mask_id: int, prefer_scale: float = 3.0):
    """Unmask remaining MASK positions independently from a categorical that prefers ``target``.

    Vocab is {0,...,V-1} with MASK last. Unmasked tokens are left unchanged.
    This is a probability kernel, not a neural net.
    """
    xtilde = jnp.asarray(xtilde)
    target = jnp.asarray(target, dtype=jnp.int32)
    L = xtilde.shape[-1]
    n_tok = vocab_size - 1
    logits = jnp.zeros((L, n_tok), dtype=jnp.float64)
    rows = jnp.arange(L, dtype=jnp.int32)
    logits = logits.at[rows, target].add(prefer_scale)
    draws = jax.random.categorical(key, logits, axis=-1).astype(xtilde.dtype)
    return jnp.where(xtilde == mask_id, draws, xtilde)
