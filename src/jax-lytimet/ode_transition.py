"""
Optional Neural-ODE latent transition, as an alternative to LyTimeT's
paper-faithful discrete residual-MLP `f_theta`.

Not part of the original paper (which explicitly uses a discrete residual
MLP -- see model.py docstring), but a natural, well-motivated upgrade for
benchmarks that are themselves governed by continuous-time ODEs (pendulum,
double pendulum, elastic pendulum, ...): instead of learning a single fixed
discrete jump z_t -> z_{t+1}, we learn a vector field

    dz/dt = g_theta(z, t)

and integrate it with an adaptive-step ODE solver (via `diffrax`) to obtain
z_{t+1} = z_t + integral_0^dt g_theta(z(s), s) ds.

Exposes the *same* call interface as `LatentTransition` (`__call__(z) ->
z_next`, `.rollout(z0, k_steps) -> (K, Dz)`), so it's a drop-in replacement
selected via `LyTimeTConfig(dynamics_type="neural_ode")`. Also exposes
`vector_field(z)` directly, used for the continuous-time Lyapunov
regularizer `dV/dt <= 0` (a tighter, solver-independent alternative to the
paper's discrete `max(0, V(z_{t+1}) - V(z_t))` penalty).
"""
from __future__ import annotations

from typing import Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


class ODEVectorField(eqx.Module):
    """g_theta(z, t): a small residual MLP giving dz/dt. Time is fed in as
    an extra scalar input so the field can (optionally) be time-varying."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, dz: int, hidden: int, *, key: PRNGKeyArray):
        """Args:
            dz: Latent state dimension.
            hidden: Hidden-layer width of the vector-field MLP.
            key: PRNG key, split internally for the three linear layers.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm = eqx.nn.LayerNorm(dz)
        self.fc1 = eqx.nn.Linear(dz + 1, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, hidden, key=k2)
        self.fc3 = eqx.nn.Linear(hidden, dz, key=k3)

    def __call__(
        self, t: Float[Array, ""], z: Float[Array, "Dz"], args=None
    ) -> Float[Array, "Dz"]:
        """diffrax-compatible vector-field call: `(t, z, args) -> dz/dt`.

        Args:
            t: Current integration time (scalar).
            z: Current latent state, shape `(Dz,)`.
            args: Unused; present for diffrax's `ODETerm` call signature.

        Returns:
            `dz/dt` at `(t, z)`, shape `(Dz,)`.
        """
        z_n = self.norm(z)
        t_feat = jnp.broadcast_to(t, (1,))
        h = jnp.concatenate([z_n, t_feat], axis=-1)
        h = jax.nn.gelu(self.fc1(h))
        h = jax.nn.gelu(self.fc2(h))
        return self.fc3(h)


class NeuralODETransition(eqx.Module):
    """Continuous-time latent transition, integrated with diffrax.

    Uses a fixed-step Tsit5 (adaptive 5th-order Runge-Kutta) integrator by
    default with `ode_solver_steps` internal steps per unit "frame time"
    `ode_dt`, which keeps compute bounded and predictable (important for a
    model meant to eventually pair with a real-time "Lite" variant) while
    still being far more accurate per step than a single Euler update.
    """

    field: ODEVectorField
    dt: float = eqx.field(static=True)
    num_internal_steps: int = eqx.field(static=True)

    def __init__(
        self,
        dz: int,
        hidden: int,
        dt: float = 1.0,
        num_internal_steps: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """Args:
            dz: Latent state dimension.
            hidden: Hidden-layer width of the vector-field MLP.
            dt: Elapsed "time" corresponding to one discrete frame step.
            num_internal_steps: Fixed number of internal solver steps used
                to integrate across each `dt`.
            key: PRNG key, forwarded to `ODEVectorField`.
        """
        self.field = ODEVectorField(dz, hidden, key=key)
        self.dt = dt
        self.num_internal_steps = num_internal_steps

    def _solve(self, z0: Float[Array, "Dz"], t_span: float) -> Float[Array, "Dz"]:
        """Integrate the vector field from `t=0` to `t=t_span` starting at `z0`.

        Args:
            z0: Initial latent state, shape `(Dz,)`.
            t_span: Total integration time.

        Returns:
            Latent state at `t=t_span`, shape `(Dz,)`.
        """
        term = diffrax.ODETerm(self.field)
        solver = diffrax.Tsit5()
        step_size = t_span / self.num_internal_steps
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_span,
            dt0=step_size,
            y0=z0,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=self.num_internal_steps + 1,
        )
        return sol.ys[-1]

    def __call__(self, z: Float[Array, "Dz"]) -> Float[Array, "Dz"]:
        """One discrete step of duration `self.dt`, matching
        LatentTransition's z_t -> z_{t+1} interface."""
        return self._solve(z, self.dt)

    def rollout(self, z0: Float[Array, "Dz"], k_steps: int) -> Float[Array, "K Dz"]:
        """Compose `k_steps` single-step solves (via `lax.scan`), each one
        identical in form to `__call__` -- i.e. each step's vector field
        sees local elapsed time in `[0, dt]`, not the trajectory's absolute
        clock time.

        This is deliberately *not* one continuous integration with
        `SaveAt` over `[0, k_steps*dt]`: since `ODEVectorField` takes `t` as
        an input feature, a single continuous solve would feed the field
        increasing absolute time (`t` in `[dt, 2dt]` for the second step,
        etc.), which would silently disagree with `__call__` composed
        repeatedly (which always resets to `t=0` each step). Scanning
        single-step solves keeps `rollout(z0, k)` and `k` repeated calls to
        `__call__` numerically consistent by construction.
        """

        def step(z, _):
            """One scan step: solve one dt-step forward via __call__."""
            z_next = self(z)
            return z_next, z_next

        _, zs = jax.lax.scan(step, z0, xs=None, length=k_steps)
        return zs

    def vector_field(self, z: Float[Array, "Dz"], t: float = 0.0) -> Float[Array, "Dz"]:
        """dz/dt at state z -- used for the continuous Lyapunov check."""
        return self.field(jnp.asarray(t), z)


def continuous_lyapunov_loss(
    transition: "NeuralODETransition",
    w: Float[Array, "Dv Ds"],
    z_seq: Float[Array, "K Dz"],
    select_idx: Optional[Int[Array, "Ds"]] = None,
) -> Float[Array, ""]:
    """Continuous-time analogue of the paper's discrete Lyapunov loss.

    For V(z~) = ||W z~||^2, dV/dt = 2 (W z~)^T (W dz~/dt). We penalize any
    positive dV/dt along the flow, i.e. any instant where the learned
    energy is *increasing* rather than contracting -- the exact, step-size-
    independent version of the paper's discrete
    `max(0, V(z_{t+1}) - V(z_t))` check.

    Args:
        transition: A `NeuralODETransition` (or any object exposing a
            compatible `vector_field(z)` method).
        w: Lyapunov energy matrix, shape `(Dv, Ds)`, defining
            `V(z~) = ||W z~||^2`.
        z_seq: Full latent states along a trajectory, shape `(K, Dz)` (the
            ODE vector field is always defined on the full `d_z`-dimensional
            state).
        select_idx: Optional indices, shape `(Ds,)`, restricting the energy
            `V` and its derivative to the selected interpretable dimensions
            `z_tilde`. If `None`, all `Dz` dimensions are used (`Ds = Dz`).

    Returns:
        Mean Lyapunov violation over the `K` states, a non-negative scalar.
    """
    dzdt = jax.vmap(lambda z: transition.vector_field(z))(z_seq)  # (K, Dz)
    z_tilde = z_seq if select_idx is None else z_seq[:, select_idx]
    dzdt_tilde = dzdt if select_idx is None else dzdt[:, select_idx]

    def per_step(
        z: Float[Array, "Ds"], dz: Float[Array, "Ds"]
    ) -> Float[Array, ""]:
        """Lyapunov violation `max(0, dV/dt)` at a single state/derivative pair."""
        wz = w @ z
        dv_dt = 2.0 * jnp.dot(wz, w @ dz)
        return jnp.maximum(0.0, dv_dt)

    violations = jax.vmap(per_step)(z_tilde, dzdt_tilde)
    return jnp.mean(violations)
