"""Sparse Fourier recovery and brute-force edit-set argmax (Alg. 1 stand-in).

Least-squares degree-<=d Walsh-Hadamard fit plus top-s coefficient truncation
stands in for ProxySPEX. Brute-force argmax over |S|<=k stands in for Gurobi
on the Appendix D.1 expansion. Exact on L=8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectral_feedback.fourier import (
    _unpack_coeffs,
    evaluate_sparse,
    fourier_inverse,
    intersection_sign,
    popcount,
)


def sample_edit_sets(key, L: int, D: int, gamma: float):
    """D independent Bernoulli(gamma) edit-sets as bitmasks (D,) and bool (D, L)."""
    bits = jax.random.bernoulli(key, gamma, (D, L))
    powers = (1 << jnp.arange(L, dtype=jnp.int32)).astype(jnp.int32)
    S_bits = (bits.astype(jnp.int32) * powers).sum(axis=-1).astype(jnp.int32)
    return S_bits, bits


def frequencies_upto_degree(L: int, d: int):
    """All subset frequencies T subset [L] with |T| <= d."""
    T = jnp.arange(2**L, dtype=jnp.int32)
    return T[popcount(T) <= d]


def _design_matrix(S_bits, T):
    """Phi[j, t] = (-1)^{|S_j ∩ T_t|}."""
    return intersection_sign(S_bits[:, None], T[None, :]).astype(jnp.float64)


def fit_truncated_fourier(S_bits, values, L: int, d: int, s: int | None = None):
    """Least squares on phi_T(S) = (-1)^{|S ∩ T|} for |T| <= d, then keep top-s by |F|.

    Returns (T_keep, F_keep, F_dense) where F_dense is length 2^L with zeros off support.
    """
    S_bits = jnp.asarray(S_bits, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.float64)
    T_all = frequencies_upto_degree(L, d)
    Phi = _design_matrix(S_bits, T_all)
    sol = jnp.linalg.lstsq(Phi, values, rcond=None)[0]
    if s is None or s >= T_all.shape[0]:
        T_keep, F_keep = T_all, sol
    else:
        order = jnp.argsort(-jnp.abs(sol))
        keep = order[:s]
        T_keep = T_all[keep]
        F_keep = sol[keep]
    F_dense = jnp.zeros((2**L,), dtype=jnp.float64)
    F_dense = F_dense.at[T_keep].set(F_keep)
    return T_keep, F_keep, F_dense


def argmax_edit_set(Fhat, k: int, L: int | None = None):
    """Brute-force S* = argmax_{|S| <= k} fhat(S) on the Appendix D.1 expansion.

    ``Fhat`` is a dense coefficient table of length 2^L, a (T, F) pair, or a dict.
    """
    if L is None:
        F_arr = jnp.asarray(Fhat) if not isinstance(Fhat, (dict, tuple, list)) else None
        if F_arr is not None and F_arr.ndim == 1:
            n = int(F_arr.shape[0])
            L = n.bit_length() - 1
        else:
            raise ValueError("L is required unless Fhat is a dense table of length 2^L")
    T, coeff = _unpack_coeffs(Fhat) if isinstance(Fhat, (dict, tuple, list)) else (
        jnp.arange(2**L, dtype=jnp.int32),
        jnp.asarray(Fhat, dtype=jnp.float64),
    )
    all_S = jnp.arange(2**L, dtype=jnp.int32)
    vals = evaluate_sparse(all_S, (T, coeff))
    vals = jnp.where(popcount(all_S) <= k, vals, -jnp.inf)
    return jnp.argmax(vals).astype(jnp.int32)


def recover_function_table(S_bits, values, L: int, d: int, s: int | None = None):
    """Reconstructed fhat table of length 2^L from sampled (S, f(S)) pairs."""
    _T, _F, F_dense = fit_truncated_fourier(S_bits, values, L, d, s)
    return fourier_inverse(F_dense)
