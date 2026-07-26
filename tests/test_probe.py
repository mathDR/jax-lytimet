import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lytimet.probe import (
    fit_linear_probe,
    amse,
    per_dimension_r2,
    rank_and_select_dimensions,
    disentanglement_consistency,
)


def test_fit_linear_probe_recovers_known_linear_map():
    """If s = z @ W_true exactly, the closed-form probe should recover
    W_true (up to numerical precision) and give near-zero AMSE."""
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (200, 5))
    w_true = jax.random.normal(jax.random.PRNGKey(1), (5, 2))
    s = z @ w_true

    w_hat = fit_linear_probe(z, s, ridge=1e-8)
    assert w_hat.shape == (5, 2)
    assert jnp.allclose(w_hat, w_true, atol=1e-2)

    err = amse(z, s, w_hat)
    assert float(err) < 1e-4


def test_fit_linear_probe_ridge_shrinks_toward_zero():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (50, 5))
    s = jax.random.normal(jax.random.PRNGKey(1), (50, 2))

    w_small_ridge = fit_linear_probe(z, s, ridge=1e-8)
    w_large_ridge = fit_linear_probe(z, s, ridge=1e3)
    assert jnp.linalg.norm(w_large_ridge) < jnp.linalg.norm(w_small_ridge)


def test_per_dimension_r2_perfect_correlation_gives_one():
    key = jax.random.PRNGKey(0)
    base = jax.random.normal(key, (100,))
    z = jnp.stack([base, jax.random.normal(jax.random.PRNGKey(1), (100,))], axis=1)
    s = base[:, None]  # s is exactly z[:, 0]

    r2 = per_dimension_r2(z, s)
    assert r2.shape == (2, 1)
    assert float(r2[0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert float(r2[1, 0]) < 0.5  # unrelated dimension should score much lower


def test_rank_and_select_dimensions_picks_the_informative_dim():
    key = jax.random.PRNGKey(0)
    informative = jax.random.normal(key, (100,))
    noise = jax.random.normal(jax.random.PRNGKey(1), (100,))
    z = jnp.stack([noise, informative, noise * 0.01], axis=1)
    s = informative[:, None]

    select_idx, scores = rank_and_select_dimensions(z, s, n_select=1)
    assert int(select_idx[0]) == 1  # the informative dimension
    assert float(scores[0]) > 0.9


def test_disentanglement_consistency_perfect_match_gives_one():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (50, 2))
    b = a  # identical trajectories
    consistency = disentanglement_consistency(a, b)
    assert consistency == pytest.approx(1.0, abs=1e-4)


def test_disentanglement_consistency_uncorrelated_gives_low_score():
    a = jax.random.normal(jax.random.PRNGKey(0), (200, 2))
    b = jax.random.normal(jax.random.PRNGKey(1), (200, 2))
    consistency = disentanglement_consistency(a, b)
    assert abs(consistency) < 0.3
