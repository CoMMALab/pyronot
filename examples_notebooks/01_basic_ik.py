"""Basic IK

Simplest Inverse Kinematics Example using PyRoNot with the HJCD-IK solver.
"""

import time
import jax
# Must be set before any JAX computation so the LM refinement phase runs in
# true float64 rather than silently downcasting to float32.
# import jax; 
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Fix the hand finger joints so IK only moves the arm.
    hand_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    fixed_joint_mask = jnp.array(
        [name in hand_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.bool_,
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_handle = server.gui.add_number("Position error (mm)", 0.0, step=1e-9, disabled=True)
    rot_error_handle = server.gui.add_number("Rotation error (rad)", 0.0, step=1e-9, disabled=True)

    target_link_index = robot.links.names.index(target_link_name)

    # JIT-compile the IK call once; subsequent calls use the compiled version.
    ik_solve = jax.jit(
        lambda pose, key, prev: robot.inverse_kinematics(
            target_link_name=target_link_name,
            target_pose=pose,
            rng_key=key,
            previous_cfg=prev,
            fixed_joint_mask=fixed_joint_mask,
        )
    )

    rng_key = jax.random.PRNGKey(0)
    # Initialise solution at the joint-range midpoint so the first call has a
    # valid warm-start reference.
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    while True:
        # Build SE(3) target from the interactive control's current pose.
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target.wxyz)),
            translation=jnp.array(ik_target.position),
        )

        # Solve IK, warm-starting from the previous solution for stability.
        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()
        solution = ik_solve(target_pose, subkey, solution)
        solution.block_until_ready()  # Wait for async JAX dispatch to finish.

        # Update timing handle.
        elapsed_time = time.perf_counter() - start_time
        timing_handle.value = elapsed_time * 1000  # Convert to milliseconds.

        # Compute positional and rotational errors.
        link_poses = robot.forward_kinematics(solution)
        actual_pose = jaxlie.SE3(link_poses[target_link_index])
        pos_error = jnp.linalg.norm(actual_pose.translation() - target_pose.translation())
        rot_error = jnp.linalg.norm((target_pose.rotation().inverse() @ actual_pose.rotation()).log())
        pos_error_handle.value = float(pos_error) * 1000  # Convert to mm.
        rot_error_handle.value = float(rot_error)

        # Update visualizer (ViserUrdf expects a numpy array).
        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
