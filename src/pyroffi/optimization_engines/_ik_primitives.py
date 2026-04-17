"""Shared primitives for IK solvers.

Provides the core residual function and constants shared by all IK solvers
in this package (_hjcd_ik, _ls_ik, etc.).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

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


def split_cuda_and_post_constraints(
    constraints: Sequence | None,
    constraint_args: Sequence | None,
    constraint_weights: Sequence[float] | None,
    collision_constraint_indices: Sequence[int] | None,
    collision_free: bool,
) -> tuple[
    tuple,
    tuple,
    Float[Array, "n_constraints"] | None,
    tuple,
    tuple,
    Float[Array, "n_constraints"] | None,
]:
    """Split constraints into CUDA-scored and post-refinement groups.

    Collision constraints are selected by index in ``collision_constraint_indices``.
    When ``collision_free`` is False those constraints are dropped entirely.
    Non-collision constraints are always retained for post-refinement.
    """
    all_fns = tuple(constraints) if constraints else ()
    all_args = tuple(constraint_args) if constraint_args is not None else ()
    if len(all_fns) == 0:
        return (), (), None, (), (), None

    if len(all_args) != len(all_fns):
        if len(all_args) == 0:
            all_args = tuple(() for _ in range(len(all_fns)))
        else:
            raise ValueError(
                "constraint_args must be None/empty or match constraints length"
            )

    if constraint_weights is None:
        all_w = jnp.ones((len(all_fns),), dtype=jnp.float32)
    else:
        all_w = jnp.array(constraint_weights, dtype=jnp.float32)
        if all_w.shape[0] != len(all_fns):
            raise ValueError("constraint_weights length must match constraints length")

    collision_idx = set(int(i) for i in (collision_constraint_indices or ()))
    for idx in collision_idx:
        if idx < 0 or idx >= len(all_fns):
            raise ValueError("collision_constraint_indices contains an out-of-range index")

    non_collision_idx = [i for i in range(len(all_fns)) if i not in collision_idx]
    if collision_free:
        cuda_idx = list(range(len(all_fns)))
    else:
        cuda_idx = non_collision_idx

    cuda_fns = tuple(all_fns[i] for i in cuda_idx)
    cuda_args = tuple(all_args[i] for i in cuda_idx)
    cuda_w = all_w[jnp.array(cuda_idx, dtype=jnp.int32)] if len(cuda_idx) > 0 else None

    post_fns = tuple(all_fns[i] for i in non_collision_idx)
    post_args = tuple(all_args[i] for i in non_collision_idx)
    post_w = (
        all_w[jnp.array(non_collision_idx, dtype=jnp.int32)]
        if len(non_collision_idx) > 0
        else None
    )

    return cuda_fns, cuda_args, cuda_w, post_fns, post_args, post_w


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
