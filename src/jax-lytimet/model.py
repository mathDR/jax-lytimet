"""
LyTimeT model components (Phase 1 architecture).

Implements, from "LyTimeT: Towards Robust and Interpretable State-Variable
Discovery" (Yu, Su, Liu, Goldfeder, Shao, Lipson; ICASSP 2026 / arXiv:2510.19716):

  * A TimeSformer-style encoder with *factorized* space-time attention
    (temporal attention over frames per patch location, then spatial
    attention over patches within a frame), operating on non-overlapping
    P x P patches with learnable spatial + temporal positional embeddings.
  * Mean-pooling of the token sequence (per frame) into a compact latent
    state z_t in R^{d_z}.
  * A lightweight deconvolutional decoder with skip connections from the
    early patch embeddings, reconstructing frames from z_t.
  * A residual-MLP latent transition function f_theta(z_t) -> z_{t+1}
    with LayerNorm + GELU, used for K-step unrolled forecasting.

All modules are plain Equinox Modules; batching is handled with jax.vmap
at the call site (see train.py) rather than baked into the modules, which
keeps the modules simple and matches idiomatic Equinox style.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LyTimeTConfig:
    image_size: int = 32          # H = W of input frames
    patch_size: int = 4           # P
    in_channels: int = 1          # grayscale by default
    dim: int = 128                # transformer token dim (d)
    depth: int = 4                # number of TimeSformer blocks (L)
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dz: int = 16                  # latent state dimension d_z
    clip_len: int = 8             # T, number of frames per training clip
    transition_hidden: int = 128
    transition_depth: int = 2     # number of residual blocks in f_theta
    dropout_rate: float = 0.0

    @property
    def num_patches_per_side(self) -> int:
        assert self.image_size % self.patch_size == 0
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2

    def lite(self) -> "LyTimeTConfig":
        """Return the LyTimeT-Lite variant: fewer heads / smaller hidden dim."""
        return dataclasses.replace(
            self,
            dim=max(32, self.dim // 2),
            num_heads=max(1, self.num_heads // 2),
            depth=max(1, self.depth - 1),
            transition_hidden=max(32, self.transition_hidden // 2),
        )


# --------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------
class PatchEmbed(eqx.Module):
    """Non-overlapping P x P conv patch embedding for a single frame."""

    proj: eqx.nn.Conv2d

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        self.proj = eqx.nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            key=key,
        )

    def __call__(self, frame: Float[Array, "C H W"]) -> Float[Array, "N D"]:
        x = self.proj(frame)  # (D, H/P, W/P)
        x = rearrange(x, "d h w -> (h w) d")
        return x


# --------------------------------------------------------------------------
# Multi-head self-attention (manual, so factorized time/space attention is
# explicit and easy to reason about / modify).
# --------------------------------------------------------------------------
class MultiHeadSelfAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim, dim, key=kq)
        self.k_proj = eqx.nn.Linear(dim, dim, key=kk)
        self.v_proj = eqx.nn.Linear(dim, dim, key=kv)
        self.out_proj = eqx.nn.Linear(dim, dim, key=ko)

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        n, d = x.shape
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        q = rearrange(q, "n (h e) -> h n e", h=self.num_heads)
        k = rearrange(k, "n (h e) -> h n e", h=self.num_heads)
        v = rearrange(v, "n (h e) -> h n e", h=self.num_heads)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_logits = jnp.einsum("hnd,hmd->hnm", q, k) * scale
        attn = jax.nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("hnm,hmd->hnd", attn, v)
        out = rearrange(out, "h n e -> n (h e)")
        return jax.vmap(self.out_proj)(out)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


class TimeSformerBlock(eqx.Module):
    """One factorized space-time attention block (divided space-time attn).

    Input/Output shape: (T, N, D) — T frames, N patches per frame, D dim.
    """

    temporal_norm: eqx.nn.LayerNorm
    temporal_attn: MultiHeadSelfAttention
    spatial_norm: eqx.nn.LayerNorm
    spatial_attn: MultiHeadSelfAttention
    mlp_norm: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        k_t, k_s, k_m = jax.random.split(key, 3)
        self.temporal_norm = eqx.nn.LayerNorm(cfg.dim)
        self.temporal_attn = MultiHeadSelfAttention(cfg.dim, cfg.num_heads, key=k_t)
        self.spatial_norm = eqx.nn.LayerNorm(cfg.dim)
        self.spatial_attn = MultiHeadSelfAttention(cfg.dim, cfg.num_heads, key=k_s)
        self.mlp_norm = eqx.nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), key=k_m)

    def __call__(self, x: Float[Array, "T N D"]) -> Float[Array, "T N D"]:
        t, n, d = x.shape

        # --- temporal attention: for each patch location n, attend over T ---
        norm_t = jax.vmap(jax.vmap(self.temporal_norm))(x)          # (T,N,D)
        tokens_by_patch = rearrange(norm_t, "t n d -> n t d")        # (N,T,D)
        attn_t = jax.vmap(self.temporal_attn)(tokens_by_patch)       # (N,T,D)
        attn_t = rearrange(attn_t, "n t d -> t n d")
        x = x + attn_t

        # --- spatial attention: for each frame t, attend over N patches ---
        norm_s = jax.vmap(jax.vmap(self.spatial_norm))(x)           # (T,N,D)
        attn_s = jax.vmap(self.spatial_attn)(norm_s)                 # (T,N,D)
        x = x + attn_s

        # --- MLP, applied per-token ---
        norm_m = jax.vmap(jax.vmap(self.mlp_norm))(x)
        mlp_out = jax.vmap(jax.vmap(self.mlp))(norm_m)
        x = x + mlp_out
        return x


class Encoder(eqx.Module):
    """TimeSformer-based encoder: clip of frames -> latent trajectory z_{1:T}."""

    cfg: LyTimeTConfig = eqx.field(static=True)
    patch_embed: PatchEmbed
    spatial_pos_emb: Float[Array, "N D"]
    temporal_pos_emb: Float[Array, "T D"]
    blocks: list
    z_head: eqx.nn.Linear

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        self.cfg = cfg
        k_patch, k_blocks, k_head = jax.random.split(key, 3)
        self.patch_embed = PatchEmbed(cfg, key=k_patch)
        self.spatial_pos_emb = jax.random.normal(
            jax.random.PRNGKey(0), (cfg.num_patches, cfg.dim)
        ) * 0.02
        self.temporal_pos_emb = jax.random.normal(
            jax.random.PRNGKey(1), (cfg.clip_len, cfg.dim)
        ) * 0.02
        block_keys = jax.random.split(k_blocks, cfg.depth)
        self.blocks = [TimeSformerBlock(cfg, key=bk) for bk in block_keys]
        self.z_head = eqx.nn.Linear(cfg.dim, cfg.dz, key=k_head)

    def __call__(
        self, clip: Float[Array, "T C H W"]
    ) -> tuple[Float[Array, "T Dz"], Float[Array, "T N D"]]:
        """Returns (z_{1:T}, patch_tokens) — the latter is kept for the
        decoder's skip connections, per Fig. 1 of the paper."""
        t = clip.shape[0]
        tokens = jax.vmap(self.patch_embed)(clip)  # (T, N, D)
        tokens = tokens + self.spatial_pos_emb[None, :, :]
        tokens = tokens + self.temporal_pos_emb[:t, None, :]

        early_tokens = tokens  # kept for decoder skip connections
        x = tokens
        for block in self.blocks:
            x = block(x)

        pooled = jnp.mean(x, axis=1)  # mean-pool over patches -> (T, D)
        z = jax.vmap(self.z_head)(pooled)  # (T, Dz)
        return z, early_tokens


# --------------------------------------------------------------------------
# Decoder: lightweight deconvolutional network with skip connections from
# the early patch embeddings.
# --------------------------------------------------------------------------
class Decoder(eqx.Module):
    cfg: LyTimeTConfig = eqx.field(static=True)
    z_to_tokens: eqx.nn.Linear
    skip_proj: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    to_image: eqx.nn.Conv2d

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        self.cfg = cfg
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        n_side = cfg.num_patches_per_side
        self.z_to_tokens = eqx.nn.Linear(cfg.dz, cfg.dim, key=k1)
        self.skip_proj = eqx.nn.Linear(cfg.dim, cfg.dim, key=k2)
        # Two upsampling stages splitting the patch_size (assumes patch_size
        # is a power of 2 with at least 2 factors of 2; falls back to a
        # single stage otherwise).
        self.deconv1 = eqx.nn.ConvTranspose2d(
            cfg.dim, cfg.dim // 2, kernel_size=4, stride=2, padding=1, key=k3
        )
        self.deconv2 = eqx.nn.ConvTranspose2d(
            cfg.dim // 2, cfg.dim // 4, kernel_size=4, stride=2, padding=1, key=k4
        )
        # Remaining upsampling factor (patch_size / 4) is handled by resizing.
        self.to_image = eqx.nn.Conv2d(
            cfg.dim // 4, cfg.in_channels, kernel_size=3, padding=1, key=k5
        )

    def __call__(
        self,
        z: Float[Array, "Dz"],
        skip_tokens: Optional[Float[Array, "N D"]] = None,
    ) -> Float[Array, "C H W"]:
        cfg = self.cfg
        n_side = cfg.num_patches_per_side
        tok = self.z_to_tokens(z)  # (D,)
        grid = jnp.broadcast_to(tok, (n_side, n_side, cfg.dim))
        if skip_tokens is not None:
            skip = jax.vmap(self.skip_proj)(skip_tokens)  # (N, D)
            skip_grid = rearrange(skip, "(h w) d -> h w d", h=n_side, w=n_side)
            grid = grid + skip_grid
        x = rearrange(grid, "h w d -> d h w")

        x = self.deconv1(x)
        x = jax.nn.gelu(x)
        x = self.deconv2(x)
        x = jax.nn.gelu(x)

        # If patch_size implies more upsampling than the two deconv stages
        # provide, do a final resize to hit the exact target resolution.
        target = cfg.image_size
        if x.shape[-1] != target:
            x = jax.image.resize(
                x, (x.shape[0], target, target), method="bilinear"
            )
        x = self.to_image(x)
        return jax.nn.sigmoid(x)


# --------------------------------------------------------------------------
# Latent transition model f_theta: residual MLP w/ LayerNorm + GELU
# --------------------------------------------------------------------------
class ResidualMLPBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(dim)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        h = self.norm(x)
        h = self.fc1(h)
        h = jax.nn.gelu(h)
        h = self.fc2(h)
        return x + h


class LatentTransition(eqx.Module):
    """f_theta(z_t) -> z_{t+1}, a residual MLP with LayerNorm + GELU."""

    blocks: list

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        keys = jax.random.split(key, cfg.transition_depth)
        self.blocks = [
            ResidualMLPBlock(cfg.dz, cfg.transition_hidden, key=k) for k in keys
        ]

    def __call__(self, z: Float[Array, "Dz"]) -> Float[Array, "Dz"]:
        for block in self.blocks:
            z = block(z)
        return z

    def rollout(self, z0: Float[Array, "Dz"], k_steps: int) -> Float[Array, "K Dz"]:
        """Recursively apply f_theta for k_steps, returning z_1..z_K (not
        including z0)."""

        def step(z, _):
            z_next = self(z)
            return z_next, z_next

        _, zs = jax.lax.scan(step, z0, xs=None, length=k_steps)
        return zs


# --------------------------------------------------------------------------
# Full Phase-1 model
# --------------------------------------------------------------------------
class LyTimeT(eqx.Module):
    cfg: LyTimeTConfig = eqx.field(static=True)
    encoder: Encoder
    decoder: Decoder
    transition: LatentTransition

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        self.cfg = cfg
        k_enc, k_dec, k_trans = jax.random.split(key, 3)
        self.encoder = Encoder(cfg, key=k_enc)
        self.decoder = Decoder(cfg, key=k_dec)
        self.transition = LatentTransition(cfg, key=k_trans)

    def encode(self, clip: Float[Array, "T C H W"]):
        return self.encoder(clip)

    def decode(self, z: Float[Array, "Dz"], skip_tokens=None):
        return self.decoder(z, skip_tokens)

    def reconstruct_clip(self, clip: Float[Array, "T C H W"]):
        z, early_tokens = self.encoder(clip)
        recon = jax.vmap(self.decoder)(z, early_tokens)
        return recon, z

    def forecast(self, z_t: Float[Array, "Dz"], k_steps: int):
        """Roll out the transition model k_steps ahead from a single state,
        decoding each predicted latent (no skip tokens available for
        future, un-encoded frames)."""
        z_future = self.transition.rollout(z_t, k_steps)  # (K, Dz)
        decode_no_skip = lambda z: self.decoder(z, None)
        x_future = jax.vmap(decode_no_skip)(z_future)
        return x_future, z_future
