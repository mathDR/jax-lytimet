"""
Compares the concatenation baseline against the control-affine structured
model for action-conditioned latent dynamics (see actions.py), directly on
the actuated pendulum's true 2D state (theta, theta_dot) -- i.e. testing
the *dynamics module* architecture question in isolation, without the
video encoder in the loop.

Trains both on a moderate torque range, then evaluates both (a) in-
distribution and (b) on torques 2-3x larger than anything seen in
training -- exactly the regime where the control-affine model's built-in
"linear in continuous action" assumption should generalize better than a
generic MLP that has to learn the scaling purely from data.

    python -m lytimet.demo_compare_actions
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .data import make_actuated_pendulum_batch
from .actions import ConcatActionTransition, ControlAffineActionTransition


DZ = 2  # (theta, theta_dot)
DISCRETE_SIZES = (2,)  # damping mode
CONTINUOUS_DIM = 1  # torque
EMBED_DIM = 4
HIDDEN = 32


def make_dataset(key, batch_size, n_steps, dt, torque_scale):
    states, modes, torques = make_actuated_pendulum_batch(
        key, batch_size, n_steps, dt=dt, torque_scale=torque_scale
    )
    d_actions = modes[..., None]  # (B, n_steps, 1) -- one discrete head
    c_actions = torques[..., None]  # (B, n_steps, 1)
    return states, d_actions, c_actions


def one_step_loss(model, states, d_actions, c_actions):
    """Teacher-forced one-step-ahead prediction loss, batched over (batch,
    time) pairs: predict state_{t+1} from (state_t, action_t)."""

    def per_sequence(states_seq, d_seq, c_seq):
        z_t = states_seq[:-1]
        z_tp1 = states_seq[1:]

        def step(z, d, c):
            return model(z, d, c)

        pred = jax.vmap(step)(z_t, d_seq, c_seq)
        return jnp.mean(jnp.sum((pred - z_tp1) ** 2, axis=-1))

    losses = jax.vmap(per_sequence)(states, d_actions, c_actions)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, states, d_actions, c_actions):
    loss, grads = eqx.filter_value_and_grad(one_step_loss)(model, states, d_actions, c_actions)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(model, key, dt, train_torque_scale, num_steps=300, batch_size=16, n_steps=15, lr=3e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    for step in range(num_steps):
        key, k_data = jax.random.split(key)
        states, d_actions, c_actions = make_dataset(k_data, batch_size, n_steps, dt, train_torque_scale)
        model, opt_state, loss = train_step(model, opt_state, optimizer, states, d_actions, c_actions)
        if step % (num_steps // 5) == 0 or step == num_steps - 1:
            print(f"    step {step:4d}  loss={float(loss):.6f}")
    return model


def evaluate(model, key, dt, torque_scale, n_eval=200, n_steps=15):
    states, d_actions, c_actions = make_dataset(key, n_eval, n_steps, dt, torque_scale)
    loss = float(one_step_loss(model, states, d_actions, c_actions))
    return loss


def main():
    dt = 0.05
    train_torque_scale = 1.0
    ood_torque_scale = 3.0  # 3x larger than anything seen in training

    key = jax.random.PRNGKey(0)
    k_concat, k_affine, k_train_concat, k_train_affine, k_eval = jax.random.split(key, 5)

    concat_model = ConcatActionTransition(
        DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN, depth=2, key=k_concat
    )
    affine_model = ControlAffineActionTransition(
        DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN, key=k_affine
    )

    print("=== Training: concatenation baseline ===")
    concat_model = train(concat_model, k_train_concat, dt, train_torque_scale)

    print("\n=== Training: control-affine model ===")
    affine_model = train(affine_model, k_train_affine, dt, train_torque_scale)

    k_eval_id, k_eval_ood = jax.random.split(k_eval)

    print("\n=== Evaluation ===")
    concat_id = evaluate(concat_model, k_eval_id, dt, train_torque_scale)
    affine_id = evaluate(affine_model, k_eval_id, dt, train_torque_scale)
    print(f"In-distribution (torque scale={train_torque_scale}):")
    print(f"    concat        one-step MSE = {concat_id:.6f}")
    print(f"    control-affine one-step MSE = {affine_id:.6f}")

    concat_ood = evaluate(concat_model, k_eval_ood, dt, ood_torque_scale)
    affine_ood = evaluate(affine_model, k_eval_ood, dt, ood_torque_scale)
    print(f"\nOut-of-distribution (torque scale={ood_torque_scale}, "
          f"{ood_torque_scale / train_torque_scale:.0f}x training range):")
    print(f"    concat        one-step MSE = {concat_ood:.6f}")
    print(f"    control-affine one-step MSE = {affine_ood:.6f}")

    print(f"\nOOD/ID error ratio: concat={concat_ood / concat_id:.2f}x, "
          f"control-affine={affine_ood / affine_id:.2f}x")
    print("(Lower ratio = better generalization to unseen action magnitudes.)")


if __name__ == "__main__":
    main()
