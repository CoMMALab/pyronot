"""IK with Collision — JAX Least-Squares Solver

Inverse Kinematics with collision avoidance using the pure-JAX LS-IK solver
and the new constraint API.

Two obstacles are demonstrated:
  - A floor half-space (z = 0).
  - A user-draggable sphere obstacle.

Constraints are added as differentiable penalty terms directly inside the
Levenberg-Marquardt normal equations, so every LM step simultaneously
minimises the task residual and drives the robot away from obstacles.

The constraint function is defined once (stable Python object) and accepts
the current sphere geometry as an explicit JAX argument (constraint_args).
Moving the sphere only changes a JAX array value — no retrace occurs.
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
    """Main function for JAX LS-IK with collision avoidance."""
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    # Fix finger joints — IK only moves the arm.
    hand_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    fixed_joint_mask = jnp.array(
        [name in hand_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.bool_,
    )

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

    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/mesh", mesh=sphere_coll_template.to_trimesh()
    )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_handle = server.gui.add_number(
        "Position error (mm)", 0.0, step=1e-9, disabled=True
    )
    coll_weight_label = server.gui.add_number(
        "Collision weight", 1e8, disabled=True
    )

    target_link_index = robot.links.names.index(target_link_name)
    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # Define the constraint function ONCE — stable Python object, no retrace.
    # The sphere geometry is passed as a dynamic JAX argument (constraint_args),
    # so moving the sphere only updates array values, never triggers retrace.
    # Smoothing radius for the collision penalty [m].  softplus approximates
    # max(0, -d) but is C-infinity smooth, giving cleaner LM gradients near
    # the constraint boundary and faster convergence.
    _COLL_EPS = 0.005

    _coll_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)

    def _collision_penalty(cfg, robot, sphere_world):
        """Differentiable collision penalty: single FK pass for both obstacles."""
        coll_geom = robot_coll.at_config(robot, cfg)
        d_plane  = _coll_vs_world(coll_geom, plane_coll.broadcast_to((1,)))
        d_sphere = _coll_vs_world(coll_geom, sphere_world.broadcast_to((1,)))
        return (
            jnp.sum(jax.nn.softplus(-d_plane / _COLL_EPS) * _COLL_EPS)
            + jnp.sum(jax.nn.softplus(-d_sphere / _COLL_EPS) * _COLL_EPS)
        )

    constraint_fns = (_collision_penalty,)
    constraint_weights = jnp.array([1e8])

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

        # ── Build IK target pose ────────────────────────────────────────────
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target_handle.wxyz)),
            translation=jnp.array(ik_target_handle.position),
        )

        # ── Solve IK with collision constraints (JAX LS solver) ─────────────
        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()

        solution = ls_ik_solve(
            robot=robot,
            target_link_index=target_link_index,
            target_pose=target_pose,
            rng_key=subkey,
            previous_cfg=solution,
            num_seeds=32,
            max_iter=60,
            fixed_joint_mask=fixed_joint_mask,
            constraint_fns=constraint_fns,
            constraint_args=(sphere_world,),
            constraint_weights=constraint_weights,
        )
        solution.block_until_ready()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_handle.value = elapsed_ms

        # ── Compute position error ────────────────────────────────────────
        link_poses = robot.forward_kinematics(solution)
        actual_pose = jaxlie.SE3(link_poses[target_link_index])
        pos_error = jnp.linalg.norm(
            actual_pose.translation() - target_pose.translation()
        )
        pos_error_handle.value = float(pos_error) * 1000  # mm

        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
