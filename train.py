"""CPU demo: Algorithm 2 on the toy reverse process, then R^2 of recovered fhat."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from spectral_feedback.diffusion import sample_reverse
from spectral_feedback.feedback import protein_model_feedback, spectral_edit_set_selection
from spectral_feedback.fourier import fourier_inverse, r_squared
from spectral_feedback.recover import fit_truncated_fourier
from spectral_feedback.toy import default_toy
from spectral_feedback.value import favg


def main():
    L = 8
    toy = default_toy(L=L, vocab_size=4)
    mask_id = toy["mask_id"]
    reward_fn = toy["reward_fn"]
    sample_fn = partial(
        sample_reverse,
        target=toy["target"],
        vocab_size=toy["vocab_size"],
        mask_id=mask_id,
        prefer_scale=2.0,
    )
    key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    x0 = sample_fn(jnp.full((L,), mask_id, dtype=jnp.int32), k0)
    r0 = float(reward_fn(x0))

    D, gamma, d, s, k, n_mc, N = 48, 0.5, 2, 8, 2, 2, 3

    def edit_select_fn(x, k_edit):
        def value_fn(S):
            return favg(x, S, k_edit, n_mc, sample_fn, reward_fn, mask_id)

        S_star, _F = spectral_edit_set_selection(
            k_edit, L, D, gamma, d, s, k, value_fn=value_fn
        )
        return S_star

    key, k_fb = jax.random.split(key)
    x_final = protein_model_feedback(k_fb, N, L, mask_id, sample_fn, edit_select_fn)
    r1 = float(reward_fn(x_final))

    key, k_tab = jax.random.split(key)
    all_S = jnp.arange(2**L, dtype=jnp.int32)
    tab_keys = jax.random.split(k_tab, 2**L)

    def one_favg(S, ks):
        return favg(x_final, S, ks, n_mc, sample_fn, reward_fn, mask_id)

    f_table = jax.vmap(one_favg)(all_S, tab_keys)
    _T, _F, F_dense = fit_truncated_fourier(all_S, f_table, L, d, s)
    fhat = fourier_inverse(F_dense)
    r2 = float(r_squared(f_table, fhat))

    print(f"initial_reward: {r0:.6f}")
    print(f"final_reward: {r1:.6f}")
    print(f"R2: {r2:.6f}")


if __name__ == "__main__":
    main()
