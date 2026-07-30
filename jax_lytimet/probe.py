"""
Phase 2, Step 1-2 of LyTimeT: linear probing and dimension ranking.

Given trained latents {z_t} and ground-truth state variables {s_t}, fit a
linear probe per ground-truth variable:

    s_hat_t^(i) = w_i^T z_t

score each latent dimension's alignment with each ground-truth variable
(via R^2, as a differentiable/closed-form proxy for the paper's mutual-
information ranking), and select the top-ranked dimensions to form the
extracted variable set z_tilde. Also implements the AMSE metric (eq. 2-3
in the paper) and a simple disentanglement check (Step 2: do z_tilde
trajectories stay consistent across different nuisance conditions?).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def fit_linear_probe(
    z: Float[Array, "T Dz"], s: Float[Array, "T Ds"], ridge: float = 1e-6
):
    """Closed-form ridge-regularized least squares probe:
        w* = argmin_w (1/T) sum_t || s_t - w^T z_t ||_2^2

    Returns w of shape (Dz, Ds) such that s_hat = z @ w.
    """
    z = jnp.asarray(z)
    s = jnp.asarray(s)
    dz = z.shape[1]
    gram = z.T @ z + ridge * jnp.eye(dz)
    w = jnp.linalg.solve(gram, z.T @ s)
    return w


def amse(z: Float[Array, "T Dz"], s: Float[Array, "T Ds"], w: Float[Array, "Dz Ds"]):
    """Analytical Mean Squared Error, eq. (2)-(3) in the paper."""
    s_hat = z @ w
    return jnp.mean(jnp.sum((s - s_hat) ** 2, axis=-1))


def per_dimension_r2(z: Float[Array, "T Dz"], s: Float[Array, "T Ds"]):
    """R^2 of each latent dimension i against each ground-truth variable j,
    used as the ranking criterion for dimension selection (a practical,
    differentiable proxy for the paper's mutual-information ranking)."""
    z = jnp.asarray(z)
    s = jnp.asarray(s)
    z_c = z - z.mean(axis=0, keepdims=True)
    s_c = s - s.mean(axis=0, keepdims=True)
    z_std = jnp.sqrt(jnp.mean(z_c ** 2, axis=0)) + 1e-8
    s_std = jnp.sqrt(jnp.mean(s_c ** 2, axis=0)) + 1e-8
    cov = (z_c.T @ s_c) / z.shape[0]  # (Dz, Ds)
    corr = cov / (z_std[:, None] * s_std[None, :])
    return corr ** 2  # (Dz, Ds), each entry is an R^2-like score


def rank_and_select_dimensions(
    z: Float[Array, "T Dz"], s: Float[Array, "T Ds"], n_select: int
):
    """Score each latent dimension by its best alignment (max R^2 across
    ground-truth variables), rank, and return the indices of the top
    `n_select` dimensions -- these form the extracted variable set z_tilde.
    """
    r2 = per_dimension_r2(z, s)  # (Dz, Ds)
    best_r2_per_dim = jnp.max(r2, axis=1)  # (Dz,)
    order = jnp.argsort(-best_r2_per_dim)
    select_idx = order[:n_select]
    scores = best_r2_per_dim[select_idx]
    return select_idx, scores


def disentanglement_consistency(
    z_tilde_a: Float[Array, "T Ds"], z_tilde_b: Float[Array, "T Ds"]
):
    """Step 2 (disentanglement validation): given the *same* underlying
    trajectory rendered under two different nuisance conditions (e.g.
    different backgrounds), measure how consistent the extracted z_tilde
    trajectories are. Returns the mean per-dimension correlation between
    the two trajectories (close to 1 => nuisance-invariant extraction)."""
    a = np.asarray(z_tilde_a)
    b = np.asarray(z_tilde_b)
    corrs = []
    for d in range(a.shape[1]):
        c = np.corrcoef(a[:, d], b[:, d])[0, 1]
        corrs.append(c)
    return float(np.mean(corrs))
