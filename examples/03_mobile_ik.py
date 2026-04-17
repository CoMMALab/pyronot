"""Basic IK with spherized Fetch robot.

Mirrors the behaviour of 01_00_basic_ik.py, but loads Fetch from the local
resources folder using the spherized URDF.
"""

import pathlib
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf


def main():
    """Main function for basic IK with the local spherized Fetch model."""
    resource_root = pathlib.Path(__file__).resolve().parent.parent / "resources"
    urdf_path = resource_root / "fetch" / "fetch_spherized.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"Spherized Fetch URDF not found: {urdf_path}")

    urdf = yourdfpy.URDF.load(str(urdf_path))
    target_link_name = "gripper_link"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0.707, 0, -0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_handle = server.gui.add_number("Position error (mm)", 0.0, step=1e-9, disabled=True)
    rot_error_handle = server.gui.add_number("Rotation error (rad)", 0.0, step=1e-9, disabled=True)

    target_link_index = robot.links.names.index(target_link_name)

    # JIT-compile once and warm-start from the previous solution each frame.
    ik_solve = jax.jit(
        lambda pose, key, prev: robot.inverse_kinematics(
            target_link_name=target_link_name,
            target_pose=pose,
            rng_key=key,
            previous_cfg=prev,
        )
    )

    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    while True:
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target.wxyz)),
            translation=jnp.array(ik_target.position),
        )

        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()
        solution = ik_solve(target_pose, subkey, solution)
        solution.block_until_ready()

        elapsed_time = time.perf_counter() - start_time
        timing_handle.value = elapsed_time * 1000

        link_poses = robot.forward_kinematics(solution)
        actual_pose = jaxlie.SE3(link_poses[target_link_index])
        pos_error = jnp.linalg.norm(actual_pose.translation() - target_pose.translation())
        rot_error = jnp.linalg.norm((target_pose.rotation().inverse() @ actual_pose.rotation()).log())
        pos_error_handle.value = float(pos_error) * 1000
        rot_error_handle.value = float(rot_error)

        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
