"""
Solves the basic IK problem with collision avoidance.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyronot as pk

def solve_collision_with_config(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    cfg: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoNot Robot.
        target_link_name: Sequence[str]. Length: num_targets.
        position: ArrayLike. Shape: (num_targets, 3), or (3,).
        wxyz: ArrayLike. Shape: (num_targets, 4), or (4,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    cfg = _solve_collision_with_config_jax(
        robot,
        coll,
        world_coll_list,
        cfg,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_collision_with_config_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    cfg: jax.Array,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)
    vars = [joint_var]

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.limit_cost(
            robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=cfg,
            weight=10.0,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
    ]
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, coll, joint_var, world_coll, 0.05, 11.0
            )
            for world_coll in world_coll_list
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]
