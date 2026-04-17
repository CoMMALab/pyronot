"""Bimanual IK with Collision — CUDA Least-Squares Solver (Baxter)

Bimanual Inverse Kinematics with collision avoidance for a Baxter robot
using the CUDA LS-IK solver.

Two end-effectors are controlled simultaneously:
  - right_gripper: primary IK target (CUDA kernel optimises this EE)
  - left_gripper:  secondary IK target (incorporated in JAX post-refinement)

Two obstacles are demonstrated:
  - A floor half-space (z = 0).
  - A user-draggable sphere obstacle.

How multi-EE + constraints work in the CUDA path
-------------------------------------------------
The CUDA kernel optimises only the first EE (right_gripper).  Both EEs are
incorporated in two JAX-side stages:

  1. Winner selection — all CUDA-returned configurations are scored with all
     EE residuals and the collision penalty.  The least penalised seed wins.

  2. Post-CUDA JAX refinement — ``constraint_refine_iters`` additional LM
     steps are run on the winner using the full multi-EE + constraint cost,
     giving the left arm a proper 6D Jacobian contribution.
"""

import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import viser
from pyroffi.collision import HalfSpace, RobotCollision, Sphere, collide
from pyroffi.optimization_engines._ls_ik import ls_ik_solve_cuda
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def main():
    """Main function for bimanual CUDA LS-IK with collision avoidance on Baxter."""
    urdf = load_robot_description("baxter_description")
    right_ee_name = "right_gripper"
    left_ee_name  = "left_gripper"

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    # Fix the head pan joint — not relevant to arm IK.
    fixed_joint_mask = jnp.array(
        [name == "head_pan" for name in robot.joints.actuated_names],
        dtype=jnp.int32,
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

    constraints        = [_collision_penalty]
    constraint_weights = [1e8]

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

        # ── Solve bimanual IK with collision constraints (CUDA LS solver) ────
        # Stage 1: CUDA kernel finds num_seeds candidates for the right arm.
        # Stage 2: JAX winner selection scores all EEs + collision penalty.
        # Stage 3: Post-CUDA JAX LM refines winner with full multi-EE + constraint cost.
        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()

        solution = ls_ik_solve_cuda(
            robot=robot,
            target_link_indices=(right_ee_idx, left_ee_idx),
            target_poses=(target_pose_right, target_pose_left),
            rng_key=subkey,
            previous_cfg=solution,
            num_seeds=256,
            fixed_joint_mask=fixed_joint_mask,
            constraints=constraints,
            constraint_args=[sphere_world],
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
