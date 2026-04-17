import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from .._robot import Robot
from ..collision import CollGeom, RobotCollision, colldist_from_sdf


@Cost.create_factory
def pose_cost(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])
    residual = (pose_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


@Cost.create_factory
def pose_cost_with_base(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_indices: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses relative to a mobile base."""
    assert target_link_indices.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    T_world_base = vals[T_world_base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)  # FK is T_base_link
    T_base_target_link = jaxlie.SE3(Ts_base_link[..., target_link_indices, :])
    T_world_target_link_actual = T_world_base @ T_base_target_link

    residual = (T_world_target_link_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


# --- Limit Costs ---


@Cost.create_factory
def limit_cost(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint limit violations."""
    joint_cfg = vals[joint_var]
    joint_cfg_eff = robot.joints.get_full_config(joint_cfg)
    residual_upper = jnp.maximum(0.0, joint_cfg_eff - robot.joints.upper_limits_all)
    residual_lower = jnp.maximum(0.0, robot.joints.lower_limits_all - joint_cfg_eff)
    return ((residual_upper + residual_lower) * weight).flatten()


@Cost.create_factory
def limit_velocity_cost(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    prev_joint_var: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint velocity limit violations."""
    joint_vel = (vals[joint_var] - vals[prev_joint_var]) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


# --- Regularization Costs ---


@Cost.create_factory
def rest_cost(
    vals: VarValues,
    joint_var: Var[Array],
    rest_pose: Array,
    weight: Array | float,
) -> Array:
    """Computes the residual biasing joints towards a rest pose."""
    return ((vals[joint_var] - rest_pose) * weight).flatten()


@Cost.create_factory
def rest_with_base_cost(
    vals: VarValues,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    rest_pose: Array,
    weight: Array | float,
) -> Array:
    """Computes the residual biasing joints and base towards rest/identity."""
    residual_joints = vals[joint_var] - rest_pose
    residual_base = vals[T_world_base_var].log()
    return (jnp.concatenate([residual_joints, residual_base]) * weight).flatten()


@Cost.create_factory
def smoothness_cost(
    vals: VarValues,
    curr_joint_var: Var[Array],
    past_joint_var: Var[Array],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint configuration differences (velocity)."""
    return ((vals[curr_joint_var] - vals[past_joint_var]) * weight).flatten()


# --- Manipulability Cost ---


def _compute_manip_yoshikawa(
    cfg: Array,
    robot: Robot,
    target_link_index: jax.Array,
) -> Array:
    """Helper: Computes manipulability measure for a single link."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[target_link_index]
    JJT = jacobian @ jacobian.T
    assert JJT.shape == (3, 3)
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))


@Cost.create_factory
def manipulability_cost(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_link_indices: Array,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing low manipulability (translation)."""
    cfg = vals[joint_var]
    if target_link_indices.ndim == 0:
        vmapped_manip = _compute_manip_yoshikawa(cfg, robot, target_link_indices)
    else:
        vmapped_manip = jax.vmap(_compute_manip_yoshikawa, in_axes=(None, None, 0))(
            cfg, robot, target_link_indices
        )
    residual = 1.0 / (vmapped_manip + 1e-6)
    return (residual * weight).flatten()


# --- Collision Costs ---


@Cost.create_factory
def self_collision_cost(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    margin: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing self-collisions below a margin."""
    cfg = vals[joint_var]
    active_distances = robot_coll.compute_self_collision_distance(robot, cfg)
    residual = colldist_from_sdf(active_distances, margin)
    return (residual * weight).flatten()


@Cost.create_factory
def world_collision_cost(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    world_geom: CollGeom,
    margin: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing world collisions below a margin."""
    cfg = vals[joint_var]
    dist_matrix = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
    residual = colldist_from_sdf(dist_matrix, margin)
    return (residual * weight).flatten()


# --- Finite Difference Costs (Velocity, Acceleration, Jerk) ---


@Cost.create_factory
def five_point_velocity_cost(
    vals: VarValues,
    robot: Robot,  # Needed for limits
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing velocity limit violations (5-point stencil)."""
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
    vel_limits = robot.joints.velocity_limits
    limit_violation = jnp.maximum(0.0, jnp.abs(velocity) - vel_limits)
    return (limit_violation * weight).flatten()


@Cost.create_factory
def five_point_acceleration_cost(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual minimizing joint acceleration (5-point stencil)."""
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    return (acceleration * weight).flatten()


@Cost.create_factory
def five_point_jerk_cost(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual minimizing joint jerk (7-point stencil)."""
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]

    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    return (jerk * weight).flatten()
