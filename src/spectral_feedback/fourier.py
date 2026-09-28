"""Walsh-Hadamard subset Fourier transform, inverse, Parseval, and R^2 (eqs. 3-5)."""

from __future__ import annotations

import jax.numpy as jnp


def popcount(x, nbits: int = 32):
    """Hamming weight of integer bitmasks."""
    x_u = jnp.asarray(x, dtype=jnp.uint32)
    shifts = jnp.arange(nbits, dtype=jnp.uint32)
    return ((x_u[..., None] >> shifts) & jnp.uint32(1)).sum(axis=-1).astype(jnp.int32)


def intersection_sign(S, T):
    """(-1)^{|S ∩ T|} with subsets encoded as bitmasks."""
    S = jnp.asarray(S, dtype=jnp.int32)
    T = jnp.asarray(T, dtype=jnp.int32)
    parity = popcount(jnp.bitwise_and(S, T)) & 1
    return 1 - 2 * parity


def _n_and_index(table):
    table = jnp.asarray(table)
    n = table.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("Fourier table length must be a power of two (2^L)")
    return table, n, jnp.arange(n, dtype=jnp.int32)


def fourier_transform(f_table):
    """Eq. (3) transform: F(T) = 2^{-L} sum_S (-1)^{|S ∩ T|} f(S)."""
    f_table, n, idx = _n_and_index(f_table)
    f_table = f_table.astype(jnp.float64)
    signs = intersection_sign(idx[None, :], idx[:, None]).astype(jnp.float64)
    return (signs * f_table[..., None, :]).sum(axis=-1) / n


def fourier_inverse(F):
    """Eq. (3) inverse: f(S) = sum_T (-1)^{|T ∩ S|} F(T)."""
    F, n, idx = _n_and_index(F)
    F = F.astype(jnp.float64)
    signs = intersection_sign(idx[:, None], idx[None, :]).astype(jnp.float64)
    return (signs * F[..., None, :]).sum(axis=-1)


def evaluate_sparse(S_bits, coeffs):
    """Appendix D.1: fhat(S) = sum_{T in A} (-1)^{|S ∩ T|} Fhat(T).

    ``coeffs`` is a dict {T: Fhat(T)}, a pair of arrays (T, Fhat), or a dense
    coefficient table of length 2^L.
    """
    S_bits = jnp.asarray(S_bits, dtype=jnp.int32)
    T, Fhat = _unpack_coeffs(coeffs)
    signs = intersection_sign(S_bits[..., None], T).astype(Fhat.dtype)
    return (signs * Fhat).sum(axis=-1)


def _unpack_coeffs(coeffs):
    if isinstance(coeffs, dict):
        T = jnp.asarray(list(coeffs.keys()), dtype=jnp.int32)
        Fhat = jnp.asarray(list(coeffs.values()), dtype=jnp.float64)
        return T, Fhat
    if isinstance(coeffs, (tuple, list)) and len(coeffs) == 2:
        T = jnp.asarray(coeffs[0], dtype=jnp.int32)
        Fhat = jnp.asarray(coeffs[1], dtype=jnp.float64)
        return T, Fhat
    Fhat = jnp.asarray(coeffs, dtype=jnp.float64)
    if Fhat.ndim != 1:
        raise ValueError("dense coefficient table must be 1-d")
    T = jnp.arange(Fhat.shape[0], dtype=jnp.int32)
    return T, Fhat


def parseval_residual(f, fhat):
    """Both sides of eq. (4) under the eq. (3) convention.

    Displayed eq. (4):
        sum_S (f(S) - fhat(S))^2 = sum_T (F(T) - Fhat(T))^2

    Eq. (3) sets F = 2^{-L} H f with Hadamard H H = 2^L I, so the identity
    that holds for the unnormalized ||.||_2 of eq. (5) is
        sum_S (f - fhat)^2 = 2^L sum_T (F - Fhat)^2.
    Those two quantities are returned (spatial, spectral).
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    fhat = jnp.asarray(fhat, dtype=jnp.float64)
    n = f.shape[-1]
    F = fourier_transform(f)
    Fhat = fourier_transform(fhat)
    spatial = jnp.sum((f - fhat) ** 2)
    spectral = n * jnp.sum((F - Fhat) ** 2)
    return spatial, spectral


def r_squared(f, fhat):
    """Eq. (5): R^2 = 1 - ||fhat - f||_2^2 / ||f - fbar||_2^2.

    ||g||_2^2 = sum_S g(S)^2 and fbar = 2^{-L} sum_S f(S).
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    fhat = jnp.asarray(fhat, dtype=jnp.float64)
    fbar = jnp.mean(f)
    num = jnp.sum((fhat - f) ** 2)
    den = jnp.sum((f - fbar) ** 2)
    return 1.0 - num / den
