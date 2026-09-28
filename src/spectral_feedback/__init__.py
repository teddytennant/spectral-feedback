"""Spectral Feedback kernel (arXiv:2609.30456) on length-L Walsh-Hadamard set functions."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from spectral_feedback.diffusion import forward_mask, sample_reverse
from spectral_feedback.feedback import protein_model_feedback, spectral_edit_set_selection
from spectral_feedback.fourier import (
    evaluate_sparse,
    fourier_inverse,
    fourier_transform,
    parseval_residual,
    r_squared,
)
from spectral_feedback.mask import remask
from spectral_feedback.recover import argmax_edit_set, fit_truncated_fourier, sample_edit_sets
from spectral_feedback.toy import planted_set_function, sequence_reward
from spectral_feedback.value import favg, fmax

__all__ = [
    "argmax_edit_set",
    "evaluate_sparse",
    "favg",
    "fit_truncated_fourier",
    "fmax",
    "forward_mask",
    "fourier_inverse",
    "fourier_transform",
    "parseval_residual",
    "planted_set_function",
    "protein_model_feedback",
    "r_squared",
    "remask",
    "sample_edit_sets",
    "sample_reverse",
    "sequence_reward",
    "spectral_edit_set_selection",
]
