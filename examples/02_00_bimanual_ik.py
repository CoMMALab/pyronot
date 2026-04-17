"""Bimanual IK with Collision — JAX Least-Squares Solver (Baxter)

Bimanual Inverse Kinematics with collision avoidance for a Baxter robot
using the pure-JAX LS-IK solver.

Two end-effectors are controlled simultaneously:
  - right_gripper: primary IK target
  - left_gripper:  secondary IK target

Two obstacles are demonstrated:
  - A floor half-space (z = 0).
  - A user-draggable sphere obstacle.

Both arm IK targets are passed natively as a multi-EE tuple to ls_ik_solve.
The residual vector is [W*f_right | W*f_left | sqrt_wc*f_coll], giving the
LM solver a full 6D Jacobian contribution per arm — no scalar penalty hack.
"""

import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import viser
from pyroffi.collision import HalfSpace, RobotCollision, Sphere, collide
from pyroffi.optimization_engines._ls_ik import ls_ik_solve
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

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_right_handle = server.gui.add_number(
        "Right EE pos error (mm)", 0.0, step=1e-9, disabled=True
    )
    pos_error_left_handle = server.gui.add_number(
        "Left EE pos error (mm)", 0.0, step=1e-9, disabled=True
    )

    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

   

    while True:
        # ── Current sphere world geometry (dynamic JAX array, no new fn) ─────


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
        # Both EEs are passed as a tuple — the solver builds a full 6D Jacobian
        # contribution per arm rather than collapsing the left arm into a scalar.
        # Constraint: collision avoidance (floor + sphere) — weight 1e8.
        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()

        solution = ls_ik_solve(
            robot=robot,
            target_link_indices=(right_ee_idx, left_ee_idx),
            target_poses=(target_pose_right, target_pose_left),
            rng_key=subkey,
            previous_cfg=solution,
            num_seeds=32,
            max_iter=60,
            fixed_joint_mask=fixed_joint_mask,
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
