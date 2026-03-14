"""Bimanual IK with Collision — JAX Least-Squares Solver (Baxter)

Bimanual Inverse Kinematics with collision avoidance for a Baxter robot
using the pure-JAX LS-IK solver.

Two end-effectors are controlled simultaneously:
  - right_gripper: primary IK target (direct task objective)
  - left_gripper:  secondary IK target (constraint penalty)

Two obstacles are demonstrated:
  - A floor half-space (z = 0).
  - A user-draggable sphere obstacle.

The right-arm IK is the primary task residual.  The left-arm IK target and
collision avoidance are added as differentiable penalty terms inside the
Levenberg-Marquardt normal equations, so every LM step simultaneously
minimises both end-effector errors and drives the robot away from obstacles.

The left-arm penalty is the weighted L2 norm of the SE(3) log-map residual
for the left gripper — when squared by the LM cost it gives:
    w * (pos_weight^2 * ||r_pos||^2 + ori_weight^2 * ||r_ori||^2)
which has the same form as the primary task cost.
"""

import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
import viser
from pyronot.collision import HalfSpace, RobotCollision, Sphere, collide
from pyronot.optimization_engines._ls_ik import ls_ik_solve
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def main():
    """Main function for bimanual JAX LS-IK with collision avoidance on Baxter."""
    urdf = load_robot_description("baxter_description")
    right_ee_name = "right_gripper"
    left_ee_name  = "left_gripper"

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    # Fix the head pan joint — not relevant to arm IK.
    fixed_joint_mask = jnp.array(
        [name == "head_pan" for name in robot.joints.actuated_names],
        dtype=jnp.bool_,
    )

    right_ee_idx = robot.links.names.index(right_ee_name)
    left_ee_idx  = robot.links.names.index(left_ee_name)

    # Static floor plane (z = 0, normal pointing up).
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # Sphere obstacle template at the origin — will be transformed each frame.
    sphere_coll_template = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # ── Viser setup ─────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    ik_target_right = server.scene.add_transform_controls(
        "/ik_target_right", scale=0.2, position=(0.65, -0.25, 0.3), wxyz=(0, 0, 1, 0)
    )
    ik_target_left = server.scene.add_transform_controls(
        "/ik_target_left", scale=0.2, position=(0.65, 0.25, 0.3), wxyz=(0, 0, 1, 0)
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.6, 0.0, 0.35)
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/mesh", mesh=sphere_coll_template.to_trimesh()
    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_right_handle = server.gui.add_number(
        "Right EE pos error (mm)", 0.0, step=1e-9, disabled=True
    )
    pos_error_left_handle = server.gui.add_number(
        "Left EE pos error (mm)", 0.0, step=1e-9, disabled=True
    )

    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # ── Constraint functions — defined once, no retrace on arg changes ──────
    _COLL_EPS = 0.005
    _coll_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)

    def _collision_penalty(cfg, robot, sphere_world):
        """Differentiable collision penalty for floor + sphere obstacle."""
        coll_geom = robot_coll.at_config(robot, cfg)
        d_plane  = _coll_vs_world(coll_geom, plane_coll.broadcast_to((1,)))
        d_sphere = _coll_vs_world(coll_geom, sphere_world.broadcast_to((1,)))
        return (
            jnp.sum(jax.nn.softplus(-d_plane / _COLL_EPS) * _COLL_EPS)
            + jnp.sum(jax.nn.softplus(-d_sphere / _COLL_EPS) * _COLL_EPS)
        )

    _LEFT_POS_W = 50.0
    _LEFT_ORI_W = 10.0

    def _left_arm_ik_penalty(cfg, robot, left_target_pose):
        """Left-arm IK residual as a differentiable scalar penalty.

        Returns the weighted L2 norm of the SE(3) log-map residual so that
        when the LM cost squares it, it yields:
            pos_weight^2 * ||r_pos||^2 + ori_weight^2 * ||r_ori||^2
        which matches the primary task cost structure.
        """
        Ts = robot.forward_kinematics(cfg)
        T_actual = jaxlie.SE3(Ts[left_ee_idx])
        res = (T_actual.inverse() @ left_target_pose).log()
        fw = jnp.concatenate([_LEFT_POS_W * res[:3], _LEFT_ORI_W * res[3:]])
        return jnp.linalg.norm(fw)

    constraint_fns    = (_collision_penalty, _left_arm_ik_penalty)
    constraint_weights = jnp.array([1e8, 1.0])

    # Initialise sphere geometry (updated each frame from the handle).
    sphere_world = sphere_coll_template.transform_from_wxyz_position(
        wxyz=np.array(sphere_handle.wxyz),
        position=np.array(sphere_handle.position),
    )

    while True:
        # ── Current sphere world geometry (dynamic JAX array, no new fn) ─────
        sphere_world = sphere_coll_template.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        # ── Build IK target poses ────────────────────────────────────────────
        target_pose_right = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target_right.wxyz)),
            translation=jnp.array(ik_target_right.position),
        )
        target_pose_left = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target_left.wxyz)),
            translation=jnp.array(ik_target_left.position),
        )

        # ── Solve bimanual IK with collision constraints (JAX LS solver) ─────
        # Primary task: right_gripper tracks target_pose_right.
        # Constraints:
        #   1. Collision avoidance (floor + sphere) — weight 1e8.
        #   2. Left arm IK (left_gripper → target_pose_left) — weight 1.0.
        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()

        solution = ls_ik_solve(
            robot=robot,
            target_link_index=right_ee_idx,
            target_pose=target_pose_right,
            rng_key=subkey,
            previous_cfg=solution,
            num_seeds=32,
            max_iter=60,
            fixed_joint_mask=fixed_joint_mask,
            constraint_fns=constraint_fns,
            constraint_args=(sphere_world, target_pose_left),
            constraint_weights=constraint_weights,
        )
        solution.block_until_ready()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_handle.value = elapsed_ms

        # ── Compute position errors for both arms ─────────────────────────
        link_poses = robot.forward_kinematics(solution)

        actual_right = jaxlie.SE3(link_poses[right_ee_idx])
        pos_error_right_handle.value = float(
            jnp.linalg.norm(actual_right.translation() - target_pose_right.translation())
        ) * 1000

        actual_left = jaxlie.SE3(link_poses[left_ee_idx])
        pos_error_left_handle.value = float(
            jnp.linalg.norm(actual_left.translation() - target_pose_left.translation())
        ) * 1000

        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
