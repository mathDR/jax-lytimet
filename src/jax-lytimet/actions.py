"""
Action-conditioned latent dynamics for LyTimeT, supporting a mix of
discrete (e.g. mode switches) and continuous (e.g. torque/force) actions.

Two architectures are provided, each in a discrete-time (residual-MLP) and
continuous-time (Neural ODE) flavor, so all four can be trained and
compared directly:

  * **Concatenation** (`ConcatActionTransition`, `ConcatActionODETransition`):
    the general-purpose baseline. Discrete actions are embedded, concatenated
    with the continuous action and the (normalized) latent state, and fed
    through an otherwise-ordinary residual MLP / vector-field MLP. Makes no
    assumption about *how* actions affect dynamics -- has to learn it purely
    from data.

  * **Control-affine** (`ControlAffineActionTransition`,
    `ControlAffineActionODETransition`): structured for physical control
    systems, which are very often affine in the continuous control input,
    e.g. a torque-driven pendulum: `theta'' = -(g/l) sin(theta) + tau / (m l^2)`
    -- additive and *linear* in `tau`. We split the update into

        drift(z, discrete)  +  B(z, discrete) @ continuous_action

    where `drift` captures the system's own (possibly mode-dependent)
    behavior and `B(z, discrete)` is a learned, state-and-mode-dependent
    "input gain" matrix multiplying the continuous action linearly. This is
    more sample-efficient and extrapolates better to action magnitudes
    outside the training range, *if* the true system really is (at least
    approximately) control-affine -- which is why both are provided, so you
    can check whether the structural assumption actually helps for your
    system rather than assuming it does.

    Discrete actions are *not* forced to enter affinely (that would be a
    strange assumption -- e.g. "gear 2" isn't "2x gear 1"): they condition
    `drift` and `B` through an embedding, exactly as in the concatenation
    baseline. Only the continuous channel gets the affine structure.

**Lyapunov-loss note** (see README): once actions exist, `L_lyap` should
be applied to the **drift term only** (i.e. `V(drift(z))`, not the full
controlled update) -- a controller is *supposed* to be able to add energy
(swing-up, etc.), and penalizing that unconditionally would fight against
correct control effort. `drift_only_vector_field` / `drift_only_step`
below expose exactly the drift term for this purpose.

All four transition classes share a common call convention:

    __call__(z, discrete_actions, continuous_action) -> z_next
    rollout(z0, discrete_actions_seq, continuous_actions_seq, k_steps) -> (K, Dz)

`discrete_actions` is an int array of shape `(n_discrete,)` (can be size 0
if you have no discrete actions); `continuous_action` is a float array of
shape `(continuous_dim,)` (can be size 0 if you have no continuous
actions). Using fixed-size (possibly zero-length) arrays rather than
`None` keeps everything `vmap`/`scan`-friendly.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from jaxtyping import Array, Float, Int, PRNGKeyArray


# --------------------------------------------------------------------------
# Action encoding: embed discrete heads, concatenate with continuous action
# --------------------------------------------------------------------------
class ActionEncoder(eqx.Module):
    """Embeds each discrete action head and concatenates with the
    continuous action vector into a single feature vector."""

    embeddings: list
    discrete_sizes: tuple = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    continuous_dim: int = eqx.field(static=True)

    def __init__(
        self,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            discrete_sizes: Number of categories for each discrete action
                head, e.g. `(3, 2)` for two discrete heads with 3 and 2
                categories respectively. Pass `()` if there are no discrete
                actions.
            continuous_dim: Dimensionality of the continuous action vector;
                pass `0` if there is no continuous action channel.
            embed_dim: Embedding width used for every discrete head.
            key: PRNG key, split internally across the discrete heads'
                embedding tables.
        """
        self.discrete_sizes = tuple(discrete_sizes)
        self.embed_dim = embed_dim
        self.continuous_dim = continuous_dim
        if self.discrete_sizes:
            keys = jax.random.split(key, len(self.discrete_sizes))
            self.embeddings = [
                eqx.nn.Embedding(n, embed_dim, key=k)
                for n, k in zip(self.discrete_sizes, keys)
            ]
        else:
            self.embeddings = []

    @property
    def feature_dim(self) -> int:
        """Total output feature width: `len(discrete_sizes) * embed_dim + continuous_dim`."""
        return len(self.discrete_sizes) * self.embed_dim + self.continuous_dim

    def __call__(
        self,
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
    ) -> Float[Array, "feature_dim"]:
        """Embed each discrete action head and concatenate with the
        continuous action.

        Args:
            discrete_actions: Discrete action indices, shape `(n_discrete,)`
                (may be size 0 if there are no discrete action heads).
            continuous_action: Continuous action vector, shape `(d_c,)` (may
                be size 0 if there is no continuous action channel).

        Returns:
            Concatenated feature vector of shape `(feature_dim,)`
            (`= len(discrete_sizes) * embed_dim + continuous_dim`; shape
            `(0,)` if both action types are absent).
        """
        parts = []
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(discrete_actions[i]))
        if self.continuous_dim > 0:
            parts.append(continuous_action)
        if not parts:
            return jnp.zeros((0,))
        return jnp.concatenate(parts, axis=-1)


def _zero_order_hold_scan(
    step_fn,
    z0: Float[Array, "Dz"],
    discrete_actions_seq: Int[Array, "K n_discrete"],
    continuous_actions_seq: Float[Array, "K d_c"],
    k_steps: int,
) -> Float[Array, "K Dz"]:
    """Shared rollout helper: scan a single-step `step_fn(z, discrete_a,
    continuous_a) -> z_next` over per-step actions (zero-order hold -- each
    action is held fixed for the duration of that step).

    Args:
        step_fn: A single-step transition callable with signature
            `(z, discrete_actions, continuous_action) -> z_next`.
        z0: Initial latent state, shape `(Dz,)`.
        discrete_actions_seq: Per-step discrete actions, shape
            `(K, n_discrete)`.
        continuous_actions_seq: Per-step continuous actions, shape
            `(K, d_c)`.
        k_steps: Number of steps to roll forward, `K`.

    Returns:
        Latent trajectory `z_1, ..., z_K` (not including `z0`), shape
        `(K, Dz)`.
    """

    def scan_body(z, actions_t):
        """One scan step: apply step_fn with that step's held-fixed
        actions, exposing z_next as both the carry and the per-step output."""
        d_a, c_a = actions_t
        z_next = step_fn(z, d_a, c_a)
        return z_next, z_next

    _, zs = jax.lax.scan(
        scan_body, z0, xs=(discrete_actions_seq, continuous_actions_seq), length=k_steps
    )
    return zs


# --------------------------------------------------------------------------
# 1. Concatenation baseline -- discrete-time (residual MLP)
# --------------------------------------------------------------------------
class _ConcatResidualBlock(eqx.Module):
    """One residual block conditioned on a concatenated action feature:
    `z + MLP(LayerNorm(z) concat a_feat)`."""

    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dz: int, da: int, hidden: int, *, key: PRNGKeyArray):
        """Args:
            dz: Latent state dimension.
            da: Action feature dimension (`ActionEncoder.feature_dim`).
            hidden: Hidden-layer width of the block's inner MLP.
            key: PRNG key, split internally for the two linear layers.
        """
        k1, k2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(dz)
        self.fc1 = eqx.nn.Linear(dz + da, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dz, key=k2)

    def __call__(self, z: Float[Array, "Dz"], a_feat: Float[Array, "Da"]) -> Float[Array, "Dz"]:
        """Args:
            z: Latent state, shape `(Dz,)`.
            a_feat: Action feature vector, shape `(Da,)`.

        Returns:
            Updated latent state, shape `(Dz,)`.
        """
        h = self.norm(z)
        h = jnp.concatenate([h, a_feat], axis=-1)
        h = jax.nn.gelu(self.fc1(h))
        h = self.fc2(h)
        return z + h


class ConcatActionTransition(eqx.Module):
    """Baseline: z_{t+1} = residual_mlp([z_t, embed(discrete_a), continuous_a])."""

    action_encoder: ActionEncoder
    blocks: list

    def __init__(
        self,
        dz: int,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        hidden: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            discrete_sizes: Category counts for each discrete action head
                (see `ActionEncoder`); `()` if none.
            continuous_dim: Continuous action dimensionality; `0` if none.
            embed_dim: Embedding width for discrete action heads.
            hidden: Hidden-layer width of each residual block.
            depth: Number of stacked `_ConcatResidualBlock`s.
            key: PRNG key, split internally for the action encoder and the
                residual block stack.
        """
        k_enc, k_blocks = jax.random.split(key)
        self.action_encoder = ActionEncoder(discrete_sizes, continuous_dim, embed_dim, key=k_enc)
        da = self.action_encoder.feature_dim
        keys = jax.random.split(k_blocks, depth)
        self.blocks = [_ConcatResidualBlock(dz, da, hidden, key=k) for k in keys]

    def __call__(
        self,
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
    ) -> Float[Array, "Dz"]:
        """Advance the latent state by one discrete step, conditioned on
        the given actions.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            continuous_action: Continuous action vector, shape `(d_c,)`.

        Returns:
            Predicted next latent state, shape `(Dz,)`.
        """
        a_feat = self.action_encoder(discrete_actions, continuous_action)
        for block in self.blocks:
            z = block(z, a_feat)
        return z

    def rollout(
        self,
        z0: Float[Array, "Dz"],
        discrete_actions_seq: Int[Array, "K n_discrete"],
        continuous_actions_seq: Float[Array, "K d_c"],
        k_steps: int,
    ) -> Float[Array, "K Dz"]:
        """Roll the transition forward `k_steps`, applying one action per step.

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            discrete_actions_seq: Per-step discrete actions, shape
                `(K, n_discrete)`.
            continuous_actions_seq: Per-step continuous actions, shape
                `(K, d_c)`.
            k_steps: Number of steps to roll forward, `K`.

        Returns:
            Latent trajectory `z_1, ..., z_K`, shape `(K, Dz)`.
        """
        return _zero_order_hold_scan(self, z0, discrete_actions_seq, continuous_actions_seq, k_steps)


# --------------------------------------------------------------------------
# 2. Concatenation baseline -- continuous-time (Neural ODE)
# --------------------------------------------------------------------------
class _ConcatActionVectorField(eqx.Module):
    """Vector field `dz/dt = g_theta(z, embed(discrete_a), continuous_a, t)`
    for the concatenation-baseline Neural ODE transition."""

    action_encoder: ActionEncoder
    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self,
        dz: int,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        hidden: int,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            discrete_sizes: Category counts for each discrete action head;
                `()` if none.
            continuous_dim: Continuous action dimensionality; `0` if none.
            embed_dim: Embedding width for discrete action heads.
            hidden: Hidden-layer width of the vector-field MLP.
            key: PRNG key, split internally for the action encoder and the
                three linear layers.
        """
        k_enc, k1, k2, k3 = jax.random.split(key, 4)
        self.action_encoder = ActionEncoder(discrete_sizes, continuous_dim, embed_dim, key=k_enc)
        da = self.action_encoder.feature_dim
        self.norm = eqx.nn.LayerNorm(dz)
        self.fc1 = eqx.nn.Linear(dz + da + 1, hidden, key=k1)  # +1 for time feature
        self.fc2 = eqx.nn.Linear(hidden, hidden, key=k2)
        self.fc3 = eqx.nn.Linear(hidden, dz, key=k3)

    def __call__(
        self,
        t: Float[Array, ""],
        z: Float[Array, "Dz"],
        args: tuple[Int[Array, "n_discrete"], Float[Array, "d_c"]],
    ) -> Float[Array, "Dz"]:
        """diffrax-compatible vector-field call: `(t, z, args) -> dz/dt`.

        Args:
            t: Current integration time (scalar).
            z: Current latent state, shape `(Dz,)`.
            args: Tuple `(discrete_actions, continuous_action)`, held fixed
                (zero-order hold) for the duration of the integration step.

        Returns:
            `dz/dt` at `(t, z)`, shape `(Dz,)`.
        """
        discrete_actions, continuous_action = args
        a_feat = self.action_encoder(discrete_actions, continuous_action)
        z_n = self.norm(z)
        t_feat = jnp.broadcast_to(t, (1,))
        h = jnp.concatenate([z_n, a_feat, t_feat], axis=-1)
        h = jax.nn.gelu(self.fc1(h))
        h = jax.nn.gelu(self.fc2(h))
        return self.fc3(h)


class ConcatActionODETransition(eqx.Module):
    """Baseline: dz/dt = g_theta(z, embed(discrete_a), continuous_a, t),
    with actions held fixed (zero-order hold) over each integration step."""

    field: _ConcatActionVectorField
    dt: float = eqx.field(static=True)
    num_internal_steps: int = eqx.field(static=True)

    def __init__(
        self,
        dz: int,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        hidden: int,
        dt: float = 1.0,
        num_internal_steps: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            discrete_sizes: Category counts for each discrete action head;
                `()` if none.
            continuous_dim: Continuous action dimensionality; `0` if none.
            embed_dim: Embedding width for discrete action heads.
            hidden: Hidden-layer width of the vector-field MLP.
            dt: Elapsed "time" per discrete step.
            num_internal_steps: Fixed number of internal solver steps used
                to integrate across each `dt`.
            key: PRNG key, forwarded to `_ConcatActionVectorField`.
        """
        self.field = _ConcatActionVectorField(
            dz, discrete_sizes, continuous_dim, embed_dim, hidden, key=key
        )
        self.dt = dt
        self.num_internal_steps = num_internal_steps

    def __call__(
        self,
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
    ) -> Float[Array, "Dz"]:
        """Integrate the vector field forward by `dt`, holding the given
        actions fixed for the duration of the step.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            continuous_action: Continuous action vector, shape `(d_c,)`.

        Returns:
            Latent state after integrating for one `dt`, shape `(Dz,)`.
        """
        term = diffrax.ODETerm(self.field)
        solver = diffrax.Tsit5()
        step_size = self.dt / self.num_internal_steps
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=self.dt, dt0=step_size, y0=z,
            args=(discrete_actions, continuous_action),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=self.num_internal_steps + 1,
        )
        return sol.ys[-1]

    def rollout(
        self,
        z0: Float[Array, "Dz"],
        discrete_actions_seq: Int[Array, "K n_discrete"],
        continuous_actions_seq: Float[Array, "K d_c"],
        k_steps: int,
    ) -> Float[Array, "K Dz"]:
        """Roll the transition forward `k_steps`, applying one action per step.

        Scanned single-dt-step solves (zero-order hold on actions, and
        consistent with `__call__`'s local-time convention -- see
        `ode_transition.py`'s `NeuralODETransition.rollout` for why a single
        continuous multi-step solve would silently disagree here).

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            discrete_actions_seq: Per-step discrete actions, shape
                `(K, n_discrete)`.
            continuous_actions_seq: Per-step continuous actions, shape
                `(K, d_c)`.
            k_steps: Number of steps to roll forward, `K`.

        Returns:
            Latent trajectory `z_1, ..., z_K`, shape `(K, Dz)`.
        """
        return _zero_order_hold_scan(self, z0, discrete_actions_seq, continuous_actions_seq, k_steps)


# --------------------------------------------------------------------------
# 3. Control-affine -- discrete-time (residual MLP drift + linear input gain)
# --------------------------------------------------------------------------
class ControlAffineActionTransition(eqx.Module):
    """z_{t+1} = z_t + drift(z_t, embed(discrete_a)) + B(z_t, embed(discrete_a)) @ continuous_a

    `drift` and `B` share a small trunk conditioned on [z, discrete_embed];
    `B` is produced as a flat vector and reshaped to (dz, continuous_dim).
    """

    action_encoder: ActionEncoder  # discrete-only (continuous_dim=0 here)
    trunk_norm: eqx.nn.LayerNorm
    trunk_fc: eqx.nn.Linear
    drift_head: eqx.nn.Linear
    input_gain_head: eqx.nn.Linear
    dz: int = eqx.field(static=True)
    continuous_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dz: int,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        hidden: int,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            discrete_sizes: Category counts for each discrete action head;
                `()` if none.
            continuous_dim: Continuous action dimensionality; `0` if none.
            embed_dim: Embedding width for discrete action heads.
            hidden: Hidden-layer width of the shared drift/input-gain trunk.
            key: PRNG key, split internally for the action encoder, trunk,
                drift head, and input-gain head.
        """
        k_enc, k_trunk, k_drift, k_gain = jax.random.split(key, 4)
        self.action_encoder = ActionEncoder(discrete_sizes, 0, embed_dim, key=k_enc)
        de = self.action_encoder.feature_dim
        self.trunk_norm = eqx.nn.LayerNorm(dz)
        self.trunk_fc = eqx.nn.Linear(dz + de, hidden, key=k_trunk)
        self.drift_head = eqx.nn.Linear(hidden, dz, key=k_drift)
        self.input_gain_head = eqx.nn.Linear(hidden, dz * max(continuous_dim, 1), key=k_gain)
        self.dz = dz
        self.continuous_dim = continuous_dim

    def _trunk(
        self, z: Float[Array, "Dz"], discrete_actions: Int[Array, "n_discrete"]
    ) -> Float[Array, "hidden"]:
        """Shared hidden representation conditioned on `[z, discrete_embed]`,
        feeding both the drift and input-gain heads.

        Args:
            z: Latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.

        Returns:
            Hidden feature vector, shape `(hidden,)`.
        """
        e = self.action_encoder(discrete_actions, jnp.zeros((0,)))
        h = jnp.concatenate([self.trunk_norm(z), e], axis=-1)
        return jax.nn.gelu(self.trunk_fc(h))

    def drift_only_step(self, z: Float[Array, "Dz"], discrete_actions: Int[Array, "n_discrete"]) -> Float[Array, "Dz"]:
        """The passive (continuous_action=0) update -- what the Lyapunov loss
        should be applied to, per the module docstring.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.

        Returns:
            Next latent state under the drift term alone, shape `(Dz,)`.
            Exactly equal to `__call__(z, discrete_actions,
            zeros(continuous_dim))`.
        """
        h = self._trunk(z, discrete_actions)
        return z + self.drift_head(h)

    def __call__(
        self,
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
    ) -> Float[Array, "Dz"]:
        """Advance the latent state by one discrete step:
        `z + drift(z, discrete) + B(z, discrete) @ continuous_action`.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            continuous_action: Continuous action vector, shape `(d_c,)`.

        Returns:
            Predicted next latent state, shape `(Dz,)`.
        """
        h = self._trunk(z, discrete_actions)
        drift = self.drift_head(h)
        if self.continuous_dim > 0:
            b_flat = self.input_gain_head(h)
            b = b_flat.reshape(self.dz, -1)[:, : self.continuous_dim]
            control_term = b @ continuous_action
        else:
            control_term = jnp.zeros(self.dz)
        return z + drift + control_term

    def rollout(
        self,
        z0: Float[Array, "Dz"],
        discrete_actions_seq: Int[Array, "K n_discrete"],
        continuous_actions_seq: Float[Array, "K d_c"],
        k_steps: int,
    ) -> Float[Array, "K Dz"]:
        """Roll the transition forward `k_steps`, applying one action per step.

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            discrete_actions_seq: Per-step discrete actions, shape
                `(K, n_discrete)`.
            continuous_actions_seq: Per-step continuous actions, shape
                `(K, d_c)`.
            k_steps: Number of steps to roll forward, `K`.

        Returns:
            Latent trajectory `z_1, ..., z_K`, shape `(K, Dz)`.
        """
        return _zero_order_hold_scan(self, z0, discrete_actions_seq, continuous_actions_seq, k_steps)


# --------------------------------------------------------------------------
# 4. Control-affine -- continuous-time (Neural ODE, control-affine vector field)
# --------------------------------------------------------------------------
class ControlAffineActionODETransition(eqx.Module):
    """dz/dt = drift(z, embed(discrete_a)) + B(z, embed(discrete_a)) @ continuous_a

    The textbook control-affine ODE form. Continuous action is held fixed
    (zero-order hold) over each integration step, as is physically standard
    for digitally-actuated control.
    """

    action_encoder: ActionEncoder  # discrete-only
    trunk_norm: eqx.nn.LayerNorm
    trunk_fc: eqx.nn.Linear
    drift_head: eqx.nn.Linear
    input_gain_head: eqx.nn.Linear
    dz: int = eqx.field(static=True)
    continuous_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_internal_steps: int = eqx.field(static=True)

    def __init__(
        self,
        dz: int,
        discrete_sizes: tuple[int, ...],
        continuous_dim: int,
        embed_dim: int,
        hidden: int,
        dt: float = 1.0,
        num_internal_steps: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            discrete_sizes: Category counts for each discrete action head;
                `()` if none.
            continuous_dim: Continuous action dimensionality; `0` if none.
            embed_dim: Embedding width for discrete action heads.
            hidden: Hidden-layer width of the shared drift/input-gain trunk.
            dt: Elapsed "time" per discrete step.
            num_internal_steps: Fixed number of internal solver steps used
                to integrate across each `dt`.
            key: PRNG key, split internally for the action encoder, trunk,
                drift head, and input-gain head.
        """
        k_enc, k_trunk, k_drift, k_gain = jax.random.split(key, 4)
        self.action_encoder = ActionEncoder(discrete_sizes, 0, embed_dim, key=k_enc)
        de = self.action_encoder.feature_dim
        self.trunk_norm = eqx.nn.LayerNorm(dz)
        self.trunk_fc = eqx.nn.Linear(dz + de + 1, hidden, key=k_trunk)  # +1 time feature
        self.drift_head = eqx.nn.Linear(hidden, dz, key=k_drift)
        self.input_gain_head = eqx.nn.Linear(hidden, dz * max(continuous_dim, 1), key=k_gain)
        self.dz = dz
        self.continuous_dim = continuous_dim
        self.dt = dt
        self.num_internal_steps = num_internal_steps

    def _trunk(
        self,
        t: Float[Array, ""],
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
    ) -> Float[Array, "hidden"]:
        """Shared hidden representation conditioned on
        `[z, discrete_embed, t]`, feeding both the drift and input-gain heads.

        Args:
            t: Current integration time (scalar).
            z: Latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.

        Returns:
            Hidden feature vector, shape `(hidden,)`.
        """
        e = self.action_encoder(discrete_actions, jnp.zeros((0,)))
        t_feat = jnp.broadcast_to(t, (1,))
        h = jnp.concatenate([self.trunk_norm(z), e, t_feat], axis=-1)
        return jax.nn.gelu(self.trunk_fc(h))

    def drift_only_vector_field(self, z: Float[Array, "Dz"], discrete_actions: Int[Array, "n_discrete"], t: float = 0.0) -> Float[Array, "Dz"]:
        """dz/dt at continuous_action=0 -- what the (continuous) Lyapunov
        check should be applied to.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            t: Current integration time (scalar; default `0.0`).

        Returns:
            `dz/dt` under the drift term alone, shape `(Dz,)`. Exactly
            equal to `vector_field(z, discrete_actions,
            zeros(continuous_dim), t)`.
        """
        h = self._trunk(jnp.asarray(t), z, discrete_actions)
        return self.drift_head(h)

    def vector_field(
        self,
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
        t: float = 0.0,
    ) -> Float[Array, "Dz"]:
        """The full control-affine vector field:
        `drift(z, discrete) + B(z, discrete) @ continuous_action`.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            continuous_action: Continuous action vector, shape `(d_c,)`.
            t: Current integration time (scalar; default `0.0`).

        Returns:
            `dz/dt` at `(z, discrete_actions, continuous_action, t)`, shape
            `(Dz,)`.
        """
        h = self._trunk(jnp.asarray(t), z, discrete_actions)
        drift = self.drift_head(h)
        if self.continuous_dim > 0:
            b_flat = self.input_gain_head(h)
            b = b_flat.reshape(self.dz, -1)[:, : self.continuous_dim]
            control_term = b @ continuous_action
        else:
            control_term = jnp.zeros(self.dz)
        return drift + control_term

    def _field_fn(
        self,
        t: Float[Array, ""],
        z: Float[Array, "Dz"],
        args: tuple[Int[Array, "n_discrete"], Float[Array, "d_c"]],
    ) -> Float[Array, "Dz"]:
        """diffrax-compatible wrapper around `vector_field`: `(t, z, args) -> dz/dt`.

        Args:
            t: Current integration time (scalar).
            z: Current latent state, shape `(Dz,)`.
            args: Tuple `(discrete_actions, continuous_action)`, held fixed
                (zero-order hold) for the duration of the integration step.

        Returns:
            `dz/dt` at `(t, z)`, shape `(Dz,)`.
        """
        discrete_actions, continuous_action = args
        return self.vector_field(z, discrete_actions, continuous_action, t)

    def __call__(
        self,
        z: Float[Array, "Dz"],
        discrete_actions: Int[Array, "n_discrete"],
        continuous_action: Float[Array, "d_c"],
    ) -> Float[Array, "Dz"]:
        """Integrate the control-affine vector field forward by `dt`,
        holding the given actions fixed for the duration of the step.

        Args:
            z: Current latent state, shape `(Dz,)`.
            discrete_actions: Discrete action indices, shape `(n_discrete,)`.
            continuous_action: Continuous action vector, shape `(d_c,)`.

        Returns:
            Latent state after integrating for one `dt`, shape `(Dz,)`.
        """
        term = diffrax.ODETerm(self._field_fn)
        solver = diffrax.Tsit5()
        step_size = self.dt / self.num_internal_steps
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=self.dt, dt0=step_size, y0=z,
            args=(discrete_actions, continuous_action),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=self.num_internal_steps + 1,
        )
        return sol.ys[-1]

    def rollout(
        self,
        z0: Float[Array, "Dz"],
        discrete_actions_seq: Int[Array, "K n_discrete"],
        continuous_actions_seq: Float[Array, "K d_c"],
        k_steps: int,
    ) -> Float[Array, "K Dz"]:
        """Roll the transition forward `k_steps`, applying one action per step.

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            discrete_actions_seq: Per-step discrete actions, shape
                `(K, n_discrete)`.
            continuous_actions_seq: Per-step continuous actions, shape
                `(K, d_c)`.
            k_steps: Number of steps to roll forward, `K`.

        Returns:
            Latent trajectory `z_1, ..., z_K`, shape `(K, Dz)`.
        """
        return _zero_order_hold_scan(self, z0, discrete_actions_seq, continuous_actions_seq, k_steps)
