import jax
import jax.numpy as jnp
import pytest

from lytimet.model import LyTimeT, LyTimeTConfig, Encoder, Decoder, LatentTransition


def test_config_derived_properties():
    cfg = LyTimeTConfig(image_size=32, patch_size=4)
    assert cfg.num_patches_per_side == 8
    assert cfg.num_patches == 64


def test_config_lite_shrinks_model():
    cfg = LyTimeTConfig(dim=128, num_heads=4, depth=4, transition_hidden=128)
    lite = cfg.lite()
    assert lite.dim < cfg.dim
    assert lite.num_heads <= cfg.num_heads
    assert lite.depth < cfg.depth
    assert lite.transition_hidden < cfg.transition_hidden


def test_encoder_output_shapes(tiny_cfg, rng_key, dummy_clip):
    encoder = Encoder(tiny_cfg, key=rng_key)
    z, tokens = encoder(dummy_clip)
    assert z.shape == (tiny_cfg.clip_len, tiny_cfg.dz)
    assert tokens.shape == (tiny_cfg.clip_len, tiny_cfg.num_patches, tiny_cfg.dim)
    assert jnp.all(jnp.isfinite(z))


def test_decoder_output_shape(tiny_cfg, rng_key):
    decoder = Decoder(tiny_cfg, key=rng_key)
    z = jax.random.normal(rng_key, (tiny_cfg.dz,))
    out = decoder(z, None)
    assert out.shape == (tiny_cfg.in_channels, tiny_cfg.image_size, tiny_cfg.image_size)
    # Decoder ends in sigmoid -> output should be in [0, 1]
    assert jnp.all(out >= 0.0) and jnp.all(out <= 1.0)


def test_decoder_with_skip_tokens_matches_shape(tiny_cfg, rng_key):
    decoder = Decoder(tiny_cfg, key=rng_key)
    z = jax.random.normal(rng_key, (tiny_cfg.dz,))
    skip = jax.random.normal(rng_key, (tiny_cfg.num_patches, tiny_cfg.dim))
    out = decoder(z, skip)
    assert out.shape == (tiny_cfg.in_channels, tiny_cfg.image_size, tiny_cfg.image_size)


def test_latent_transition_shape_and_residual_structure(tiny_cfg, rng_key):
    transition = LatentTransition(tiny_cfg, key=rng_key)
    z = jax.random.normal(rng_key, (tiny_cfg.dz,))
    z_next = transition(z)
    assert z_next.shape == z.shape
    assert jnp.all(jnp.isfinite(z_next))


def test_latent_transition_rollout_shape(tiny_cfg, rng_key):
    transition = LatentTransition(tiny_cfg, key=rng_key)
    z0 = jax.random.normal(rng_key, (tiny_cfg.dz,))
    traj = transition.rollout(z0, 4)
    assert traj.shape == (4, tiny_cfg.dz)


@pytest.mark.parametrize("dynamics_type", ["residual_mlp", "neural_ode"])
def test_lytimet_forward_pass_both_dynamics(tiny_cfg, rng_key, dummy_clip, dynamics_type):
    cfg = tiny_cfg if dynamics_type == "residual_mlp" else _ode_variant(tiny_cfg)
    model = LyTimeT(cfg, key=rng_key)

    recon, z = model.reconstruct_clip(dummy_clip)
    assert recon.shape == dummy_clip.shape
    assert z.shape == (cfg.clip_len, cfg.dz)
    assert jnp.all(jnp.isfinite(recon))
    assert jnp.all(jnp.isfinite(z))

    x_future, z_future = model.forecast(z[0], 3)
    assert x_future.shape == (3, cfg.in_channels, cfg.image_size, cfg.image_size)
    assert z_future.shape == (3, cfg.dz)
    assert jnp.all(jnp.isfinite(x_future))


def _ode_variant(cfg: LyTimeTConfig) -> LyTimeTConfig:
    import dataclasses

    return dataclasses.replace(cfg, dynamics_type="neural_ode", ode_solver_steps=2, ode_dt=1.0)


def test_unknown_dynamics_type_raises(tiny_cfg, rng_key):
    import dataclasses

    bad_cfg = dataclasses.replace(tiny_cfg, dynamics_type="not_a_real_type")
    with pytest.raises(ValueError):
        LyTimeT(bad_cfg, key=rng_key)


def test_model_is_valid_equinox_pytree(tiny_model):
    # eqx.filter/partition should not choke on the model; array leaves should
    # all be finite. This guards against accidentally introducing NaNs at
    # initialization (e.g. via a bad init scale).
    import equinox as eqx

    leaves = jax.tree_util.tree_leaves(eqx.filter(tiny_model, eqx.is_array))
    assert len(leaves) > 0
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))
