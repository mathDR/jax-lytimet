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
    """Hyperparameters for the full LyTimeT model (encoder + decoder +
    latent transition).

    Attributes:
        image_size: Height/width `H = W` of input frames (frames are assumed
            square).
        patch_size: Side length `P` of each non-overlapping square patch fed
            to the encoder; must evenly divide `image_size`.
        in_channels: Number of image channels (1 for grayscale, 3 for RGB).
        dim: TimeSformer token embedding dimension `d`.
        depth: Number of stacked `TimeSformerBlock`s, `L`.
        num_heads: Number of attention heads in each `MultiHeadSelfAttention`
            (temporal and spatial); must evenly divide `dim`.
        mlp_ratio: Hidden-layer expansion factor for each block's MLP
            (hidden size = `dim * mlp_ratio`).
        dz: Latent state dimension `d_z` (the compact per-frame state `z_t`).
        clip_len: Number of frames `T` per training clip (also the length of
            the learned temporal positional embedding table).
        transition_hidden: Hidden width of the latent transition network
            (residual-MLP blocks or, for `dynamics_type="neural_ode"`, the
            vector-field MLP).
        transition_depth: Number of residual blocks in the residual-MLP
            transition (`dynamics_type="residual_mlp"` only).
        dropout_rate: Reserved for future use; currently unused by any
            module in this package.
        dynamics_type: Either `"residual_mlp"` (the paper's default: a
            discrete residual MLP) or `"neural_ode"` (an optional,
            non-paper Neural ODE transition, see `ode_transition.py`).
        ode_solver_steps: Number of fixed internal solver steps per unit
            "frame time" (`dynamics_type="neural_ode"` only).
        ode_dt: The elapsed "time" corresponding to one discrete frame step
            (`dynamics_type="neural_ode"` only).
    """

    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 1
    dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dz: int = 16
    clip_len: int = 8
    transition_hidden: int = 128
    transition_depth: int = 2
    dropout_rate: float = 0.0

    dynamics_type: str = "residual_mlp"
    ode_solver_steps: int = 4
    ode_dt: float = 1.0

    @property
    def num_patches_per_side(self) -> int:
        """Number of patches along one side of the frame, `image_size / patch_size`."""
        assert self.image_size % self.patch_size == 0
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total number of patches per frame, `num_patches_per_side ** 2`."""
        return self.num_patches_per_side ** 2

    def lite(self) -> "LyTimeTConfig":
        """Return the LyTimeT-Lite variant: fewer attention heads, shallower
        encoder, and a smaller transition hidden width, for cheaper/faster
        (e.g. near-real-time) inference at some accuracy cost.

        Returns:
            A new `LyTimeTConfig` with `dim`, `num_heads`, `depth`, and
            `transition_hidden` reduced (each floored at a sane minimum);
            all other fields are copied unchanged from `self`.
        """
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
        """Build the patch-embedding convolution.

        Args:
            cfg: Model configuration; uses `in_channels`, `dim`, and
                `patch_size`.
            key: PRNG key used to initialize the convolution's parameters.
        """
        self.proj = eqx.nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            key=key,
        )

    def __call__(self, frame: Float[Array, "C H W"]) -> Float[Array, "N D"]:
        """Embed one frame into a sequence of patch tokens.

        Args:
            frame: A single image of shape `(C, H, W)`.

        Returns:
            Patch token sequence of shape `(N, D)`, where
            `N = (H/P) * (W/P)` and `D = cfg.dim`.
        """
        x = self.proj(frame)  # (D, H/P, W/P)
        x = rearrange(x, "d h w -> (h w) d")
        return x


# --------------------------------------------------------------------------
# Multi-head self-attention (manual, so factorized time/space attention is
# explicit and easy to reason about / modify).
# --------------------------------------------------------------------------
class MultiHeadSelfAttention(eqx.Module):
    """Standard scaled dot-product multi-head self-attention over a
    sequence of tokens (no causal masking; used for both the temporal and
    spatial attention passes in `TimeSformerBlock`)."""

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        """Args:
            dim: Token embedding dimension; must be evenly divisible by
                `num_heads`.
            num_heads: Number of attention heads.
            key: PRNG key, split internally to initialize the four linear
                projections (query, key, value, output).
        """
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim, dim, key=kq)
        self.k_proj = eqx.nn.Linear(dim, dim, key=kk)
        self.v_proj = eqx.nn.Linear(dim, dim, key=kv)
        self.out_proj = eqx.nn.Linear(dim, dim, key=ko)

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Apply self-attention over the `N` tokens in `x`.

        Args:
            x: Token sequence of shape `(N, D)`.

        Returns:
            Attended token sequence, same shape `(N, D)`.
        """
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
    """Two-layer feed-forward network with a GELU nonlinearity (the MLP
    sub-block used inside each `TimeSformerBlock`)."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key: PRNGKeyArray):
        """Args:
            dim: Input and output feature dimension.
            hidden: Hidden-layer width.
            key: PRNG key, split internally for the two linear layers.
        """
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        """Args:
            x: Input feature vector of shape `(D,)`.

        Returns:
            Output feature vector, same shape `(D,)`.
        """
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
        """Args:
            cfg: Model configuration; uses `dim`, `num_heads`, and `mlp_ratio`.
            key: PRNG key, split internally for the temporal attention,
                spatial attention, and MLP sub-modules.
        """
        k_t, k_s, k_m = jax.random.split(key, 3)
        self.temporal_norm = eqx.nn.LayerNorm(cfg.dim)
        self.temporal_attn = MultiHeadSelfAttention(cfg.dim, cfg.num_heads, key=k_t)
        self.spatial_norm = eqx.nn.LayerNorm(cfg.dim)
        self.spatial_attn = MultiHeadSelfAttention(cfg.dim, cfg.num_heads, key=k_s)
        self.mlp_norm = eqx.nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), key=k_m)

    def __call__(self, x: Float[Array, "T N D"]) -> Float[Array, "T N D"]:
        """Apply temporal attention (per patch location, across frames),
        then spatial attention (per frame, across patches), then a
        per-token MLP -- each with a residual connection.

        Args:
            x: Token grid of shape `(T, N, D)` -- `T` frames, `N` patches
                per frame, `D` embedding dim.

        Returns:
            Updated token grid, same shape `(T, N, D)`.
        """
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
    blocks: list[TimeSformerBlock]
    z_head: eqx.nn.Linear

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        """Args:
            cfg: Model configuration.
            key: PRNG key, split internally for the patch embedding,
                `TimeSformerBlock` stack, and latent projection head.
        """
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
        """Encode a clip of frames into a per-frame latent trajectory.

        Args:
            clip: Frame sequence of shape `(T, C, H, W)`.

        Returns:
            A tuple `(z, patch_tokens)`:
              - `z`: latent state per frame, shape `(T, Dz)`.
              - `patch_tokens`: the pre-attention patch embeddings (with
                positional embeddings added) of shape `(T, N, D)`, kept for
                the decoder's skip connections per Fig. 1 of the paper.
        """
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
    """Lightweight deconvolutional decoder: latent state z_t -> reconstructed
    frame, with optional skip connections from the encoder's early patch
    tokens (available when decoding an already-encoded frame; unavailable
    when decoding a purely forecasted future latent)."""

    cfg: LyTimeTConfig = eqx.field(static=True)
    z_to_tokens: eqx.nn.Linear
    skip_proj: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    to_image: eqx.nn.Conv2d

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        """Args:
            cfg: Model configuration; uses `dz`, `dim`, `in_channels`,
                `num_patches_per_side`, and `image_size`.
            key: PRNG key, split internally for the five sub-layers
                (latent projection, skip projection, two deconvolutions,
                and the final RGB/grayscale projection).
        """
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
        """Decode one latent state into a reconstructed frame.

        Args:
            z: Latent state of shape `(Dz,)`.
            skip_tokens: Optional encoder patch tokens of shape `(N, D)` to
                add as a skip connection (pass `None` when decoding a
                forecasted, never-encoded latent).

        Returns:
            Reconstructed frame of shape `(C, H, W)`, with pixel values in
            `[0, 1]` (final activation is a sigmoid).
        """
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
    """One residual MLP block: `z + MLP(LayerNorm(z))`, with GELU."""

    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key: PRNGKeyArray):
        """Args:
            dim: Input/output feature dimension (the latent size `d_z`).
            hidden: Hidden-layer width of the block's inner MLP.
            key: PRNG key, split internally for the two linear layers.
        """
        k1, k2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(dim)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        """Args:
            x: Input vector of shape `(D,)`.

        Returns:
            `x + MLP(LayerNorm(x))`, same shape `(D,)`.
        """
        h = self.norm(x)
        h = self.fc1(h)
        h = jax.nn.gelu(h)
        h = self.fc2(h)
        return x + h


class LatentTransition(eqx.Module):
    """f_theta(z_t) -> z_{t+1}, a residual MLP with LayerNorm + GELU.

    This is the paper's default latent transition model
    (`LyTimeTConfig.dynamics_type == "residual_mlp"`). See
    `ode_transition.NeuralODETransition` for the alternative, continuous-time
    Neural ODE transition, which shares this class's call interface.
    """

    blocks: list[ResidualMLPBlock]

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        """Args:
            cfg: Model configuration; uses `dz`, `transition_hidden`, and
                `transition_depth`.
            key: PRNG key, split internally across the `transition_depth`
                residual blocks.
        """
        keys = jax.random.split(key, cfg.transition_depth)
        self.blocks = [
            ResidualMLPBlock(cfg.dz, cfg.transition_hidden, key=k) for k in keys
        ]

    def __call__(self, z: Float[Array, "Dz"]) -> Float[Array, "Dz"]:
        """Advance the latent state by one discrete step.

        Args:
            z: Current latent state, shape `(Dz,)`.

        Returns:
            Predicted next latent state `z_{t+1}`, shape `(Dz,)`.
        """
        for block in self.blocks:
            z = block(z)
        return z

    def rollout(self, z0: Float[Array, "Dz"], k_steps: int) -> Float[Array, "K Dz"]:
        """Recursively apply f_theta for k_steps, returning z_1..z_K (not
        including z0).

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            k_steps: Number of steps to roll forward, `K`.

        Returns:
            Latent trajectory `z_1, ..., z_K`, shape `(K, Dz)`.
        """

        def step(z, _):
            """One scan step: apply f_theta once, exposing z_next as both
            the carry and the per-step output for jax.lax.scan."""
            z_next = self(z)
            return z_next, z_next

        _, zs = jax.lax.scan(step, z0, xs=None, length=k_steps)
        return zs


# --------------------------------------------------------------------------
# Full Phase-1 model
# --------------------------------------------------------------------------
class LyTimeT(eqx.Module):
    """The full Phase-1 LyTimeT model: TimeSformer `Encoder` + `Decoder` +
    a latent transition module (`LatentTransition` or, optionally,
    `ode_transition.NeuralODETransition`)."""

    cfg: LyTimeTConfig = eqx.field(static=True)
    encoder: Encoder
    decoder: Decoder
    transition: eqx.Module  # LatentTransition or NeuralODETransition -- same interface

    def __init__(self, cfg: LyTimeTConfig, *, key: PRNGKeyArray):
        """Args:
            cfg: Model configuration; `cfg.dynamics_type` selects which
                transition module is constructed (`"residual_mlp"` or
                `"neural_ode"`).
            key: PRNG key, split internally for the encoder, decoder, and
                transition sub-modules.

        Raises:
            ValueError: If `cfg.dynamics_type` is not one of
                `"residual_mlp"` or `"neural_ode"`.
        """
        self.cfg = cfg
        k_enc, k_dec, k_trans = jax.random.split(key, 3)
        self.encoder = Encoder(cfg, key=k_enc)
        self.decoder = Decoder(cfg, key=k_dec)

        if cfg.dynamics_type == "residual_mlp":
            self.transition = LatentTransition(cfg, key=k_trans)
        elif cfg.dynamics_type == "neural_ode":
            # Imported lazily so `diffrax` is only required if you actually
            # opt into the neural_ode dynamics.
            from .ode_transition import NeuralODETransition

            self.transition = NeuralODETransition(
                dz=cfg.dz,
                hidden=cfg.transition_hidden,
                dt=cfg.ode_dt,
                num_internal_steps=cfg.ode_solver_steps,
                key=k_trans,
            )
        else:
            raise ValueError(f"Unknown dynamics_type: {cfg.dynamics_type!r}")

    def encode(
        self, clip: Float[Array, "T C H W"]
    ) -> tuple[Float[Array, "T Dz"], Float[Array, "T N D"]]:
        """Encode a clip into its latent trajectory and early patch tokens.

        Args:
            clip: Frame sequence of shape `(T, C, H, W)`.

        Returns:
            Same as `Encoder.__call__`: `(z, early_tokens)`.
        """
        return self.encoder(clip)

    def decode(
        self,
        z: Float[Array, "Dz"],
        skip_tokens: Optional[Float[Array, "N D"]] = None,
    ) -> Float[Array, "C H W"]:
        """Decode a single latent state into a reconstructed frame.

        Args:
            z: Latent state, shape `(Dz,)`.
            skip_tokens: Optional encoder patch tokens for the skip
                connection; see `Decoder.__call__`.

        Returns:
            Reconstructed frame, shape `(C, H, W)`.
        """
        return self.decoder(z, skip_tokens)

    def reconstruct_clip(
        self, clip: Float[Array, "T C H W"]
    ) -> tuple[Float[Array, "T C H W"], Float[Array, "T Dz"]]:
        """Encode a clip and decode every frame back (using skip connections
        from that same frame's patch tokens).

        Args:
            clip: Frame sequence of shape `(T, C, H, W)`.

        Returns:
            A tuple `(recon, z)`:
              - `recon`: reconstructed frames, shape `(T, C, H, W)`.
              - `z`: the encoded latent trajectory, shape `(T, Dz)`.
        """
        z, early_tokens = self.encoder(clip)
        recon = jax.vmap(self.decoder)(z, early_tokens)
        return recon, z

    def forecast(
        self, z_t: Float[Array, "Dz"], k_steps: int
    ) -> tuple[Float[Array, "K C H W"], Float[Array, "K Dz"]]:
        """Roll out the transition model k_steps ahead from a single state,
        decoding each predicted latent (no skip tokens available for
        future, un-encoded frames).

        Args:
            z_t: Starting latent state, shape `(Dz,)`.
            k_steps: Number of future steps to forecast, `K`.

        Returns:
            A tuple `(x_future, z_future)`:
              - `x_future`: decoded future frames, shape `(K, C, H, W)`.
              - `z_future`: the rolled-out latent trajectory, shape `(K, Dz)`.
        """
        z_future = self.transition.rollout(z_t, k_steps)  # (K, Dz)
        decode_no_skip = lambda z: self.decoder(z, None)
        x_future = jax.vmap(decode_no_skip)(z_future)
        return x_future, z_future
