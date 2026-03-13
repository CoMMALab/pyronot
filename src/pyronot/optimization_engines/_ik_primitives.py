"""Shared primitives for IK solvers.

Provides the core residual function and constants shared by all IK solvers
in this package (_hjcd_ik, _ls_ik, etc.).
"""

from __future__ import annotations

import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot

# Step-size candidates for the vectorised LM line search.
# All five are evaluated in parallel via vmap; the best is kept.
_LS_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025])


@functools.partial(jax.jit, static_argnames=("target_link_index",))
def _ik_residual(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> Float[Array, "6"]:
    """SE(3) log-map residual.  Layout: [pos(3), ori(3)]."""
    Ts_world_link = robot.forward_kinematics(cfg)
    T_actual = jaxlie.SE3(Ts_world_link[target_link_index])
    return (T_actual.inverse() @ target_pose).log()


@jax.jit
def _adaptive_weights(f: Float[Array, "6"]) -> Float[Array, "6"]:
    """Adaptive position / orientation balance weights.

    When position error dominates, orientation residuals are down-weighted
    so the solver focuses on closing the translational gap first.  The scale
    is clipped to [0.05, 1.0] so orientation never loses all influence.
    """
    pos_err = jnp.linalg.norm(f[:3]) + 1e-8
    ori_err = jnp.linalg.norm(f[3:]) + 1e-8
    ori_scale = jnp.clip(pos_err / ori_err, 0.05, 1.0)
    return jnp.concatenate([jnp.ones(3), jnp.full(3, ori_scale)])
