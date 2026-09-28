"""Lock the displayed equations of arXiv:2609.30456, not a rewrite."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from spectral_feedback.diffusion import sample_reverse
from spectral_feedback.feedback import protein_model_feedback, spectral_edit_set_selection
from spectral_feedback.fourier import (
    evaluate_sparse,
    fourier_inverse,
    fourier_transform,
    parseval_residual,
    r_squared,
)
from spectral_feedback.mask import remask
from spectral_feedback.recover import argmax_edit_set
from spectral_feedback.toy import PLANTED_F, PLANTED_T, default_toy, planted_set_function
from spectral_feedback.value import favg, fmax

L = 8
N = 2**L
ATOL = 1e-5


def _np(x):
    return np.asarray(x, dtype=np.float64)


def _sign(S: int, T: int) -> float:
    return float((-1) ** bin((S & T) & 0xFFFFFFFF).count("1"))


def explicit_transform(f: np.ndarray) -> np.ndarray:
    """Eq. (3) transform written as a double sum over subsets."""
    n = f.shape[0]
    F = np.zeros(n, dtype=np.float64)
    for T in range(n):
        acc = 0.0
        for S in range(n):
            acc += _sign(S, T) * f[S]
        F[T] = acc / n
    return F


def explicit_inverse(F: np.ndarray) -> np.ndarray:
    """Eq. (3) inverse written as a sum over frequencies."""
    n = F.shape[0]
    f = np.zeros(n, dtype=np.float64)
    for S in range(n):
        acc = 0.0
        for T in range(n):
            acc += _sign(T, S) * F[T]
        f[S] = acc
    return f


def test_remask_eq1_only_S_becomes_mask():
    x = jnp.arange(L, dtype=jnp.int32) + 10
    mask_id = 99
    S_bool = jnp.array([True, False, True, False, False, False, False, False])
    xtilde = remask(x, S_bool, mask_id)
    assert int(xtilde[0]) == mask_id
    assert int(xtilde[2]) == mask_id
    for i in range(L):
        if i not in (0, 2):
            assert int(xtilde[i]) == int(x[i])
    xtilde_bits = remask(x, jnp.int32(0b00000101), mask_id)
    np.testing.assert_array_equal(_np(xtilde), _np(xtilde_bits))


def test_fourier_transform_inverse_recover_random_table():
    rng = np.random.default_rng(0)
    f = rng.standard_normal(N)
    F = fourier_transform(jnp.asarray(f))
    f_rec = fourier_inverse(F)
    np.testing.assert_allclose(_np(f_rec), f, atol=ATOL, rtol=0.0)


def test_fourier_transform_matches_displayed_eq3():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(N)
    F = _np(fourier_transform(jnp.asarray(f)))
    np.testing.assert_allclose(F, explicit_transform(f), atol=ATOL, rtol=0.0)


def test_parseval_eq4_truncated_top8():
    rng = np.random.default_rng(2)
    f = jnp.asarray(rng.standard_normal(N), dtype=jnp.float64)
    F = fourier_transform(f)
    top = jnp.argsort(-jnp.abs(F))[:8]
    Fhat = jnp.zeros_like(F).at[top].set(F[top])
    fhat = fourier_inverse(Fhat)
    left, right = parseval_residual(f, fhat)
    np.testing.assert_allclose(_np(left), _np(right), atol=ATOL, rtol=0.0)
    spatial = float(jnp.sum((f - fhat) ** 2))
    np.testing.assert_allclose(_np(left), spatial, atol=ATOL, rtol=0.0)


def test_r2_identity_is_one_mean_is_zero():
    rng = np.random.default_rng(3)
    f = jnp.asarray(rng.standard_normal(N), dtype=jnp.float64)
    np.testing.assert_allclose(_np(r_squared(f, f)), 1.0, atol=ATOL, rtol=0.0)
    fbar = jnp.full_like(f, jnp.mean(f))
    np.testing.assert_allclose(_np(r_squared(f, fbar)), 0.0, atol=ATOL, rtol=0.0)


def test_inverse_formula_at_one_S():
    rng = np.random.default_rng(4)
    f = rng.standard_normal(N)
    F = fourier_transform(jnp.asarray(f))
    S = 19
    acc = 0.0
    for T in range(N):
        acc += _sign(T, S) * float(F[T])
    np.testing.assert_allclose(acc, float(fourier_inverse(F)[S]), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        acc, float(evaluate_sparse(S, (jnp.arange(N), F))), atol=ATOL, rtol=0.0
    )
    np.testing.assert_allclose(acc, explicit_inverse(_np(F))[S], atol=ATOL, rtol=0.0)


def test_planted_sparse_exact_transform():
    f, F_true = planted_set_function(L)
    F = fourier_transform(f)
    np.testing.assert_allclose(_np(F), _np(F_true), atol=1e-6, rtol=0.0)
    for T in range(N):
        if T in PLANTED_T:
            idx = PLANTED_T.index(T)
            np.testing.assert_allclose(float(F[T]), PLANTED_F[idx], atol=1e-6, rtol=0.0)
        else:
            np.testing.assert_allclose(float(F[T]), 0.0, atol=1e-6, rtol=0.0)


def test_argmax_edit_set_k2_matches_brute_force():
    f, F = planted_set_function(L)
    S_star = int(argmax_edit_set(F, k=2, L=L))
    assert bin(S_star).count("1") <= 2
    all_S = np.arange(N)
    vals = _np(f).copy()
    vals[np.array([bin(int(s)).count("1") > 2 for s in all_S])] = -np.inf
    best = float(np.max(vals))
    np.testing.assert_allclose(float(f[S_star]), best, atol=1e-6, rtol=0.0)
    fhat_star = float(evaluate_sparse(S_star, F))
    np.testing.assert_allclose(fhat_star, best, atol=1e-6, rtol=0.0)


def test_favg_fmax_identical_samples_equal_r():
    z = jnp.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32)

    def sample_fn(xtilde, key):
        return z

    def reward_fn(seq):
        return jnp.sum(seq.astype(jnp.float64))

    x = jnp.zeros((L,), dtype=jnp.int32)
    S = jnp.int32(0b00110011)
    key = jax.random.PRNGKey(0)
    r_z = float(reward_fn(z))
    np.testing.assert_allclose(
        float(favg(x, S, key, 5, sample_fn, reward_fn, mask_id=3)), r_z, atol=1e-6
    )
    np.testing.assert_allclose(
        float(fmax(x, S, key, 5, sample_fn, reward_fn, mask_id=3)), r_z, atol=1e-6
    )


def test_alg2_n2_returns_unmasked_length_L():
    toy = default_toy(L=L, vocab_size=4)
    sample_fn = partial(
        sample_reverse,
        target=toy["target"],
        vocab_size=toy["vocab_size"],
        mask_id=toy["mask_id"],
        prefer_scale=2.0,
    )

    def edit_select_fn(x, key):
        def value_fn(S):
            return favg(x, S, key, 1, sample_fn, toy["reward_fn"], toy["mask_id"])

        S_star, _F = spectral_edit_set_selection(
            key, L, D=8, gamma=0.5, d=1, s=4, k=2, value_fn=value_fn
        )
        return S_star

    x = protein_model_feedback(
        jax.random.PRNGKey(1),
        N=2,
        L=L,
        mask_id=toy["mask_id"],
        sample_fn=sample_fn,
        edit_select_fn=edit_select_fn,
    )
    assert x.shape == (L,)
    assert int(jnp.all(x != toy["mask_id"]))
    assert int(jnp.all((x >= 0) & (x < toy["vocab_size"])))


def test_spectral_step_on_planted_matches_best_k():
    f, _F = planted_set_function(L)
    k = 2
    S_all = jnp.arange(N, dtype=jnp.int32)
    S_star, Fhat = spectral_edit_set_selection(
        jax.random.PRNGKey(2),
        L,
        D=N,
        gamma=0.5,
        d=2,
        s=4,
        k=k,
        S_bits=S_all,
        values=f,
    )
    S_star = int(S_star)
    assert bin(S_star).count("1") <= k
    all_S = np.arange(N)
    vals = _np(f).copy()
    vals[np.array([bin(int(s)).count("1") > k for s in all_S])] = -np.inf
    best = float(np.max(vals))
    np.testing.assert_allclose(float(f[S_star]), best, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(float(evaluate_sparse(S_star, Fhat)), best, atol=1e-6, rtol=0.0)


def test_evaluate_sparse_dict_matches_dense():
    f, F = planted_set_function(L)
    coeffs = {int(T): float(PLANTED_F[i]) for i, T in enumerate(PLANTED_T)}
    S = 13
    np.testing.assert_allclose(
        float(evaluate_sparse(S, coeffs)), float(f[S]), atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        float(evaluate_sparse(S, F)), float(f[S]), atol=1e-6, rtol=0.0
    )
