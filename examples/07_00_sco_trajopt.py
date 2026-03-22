"""Trajectory Optimization

Trajectory optimization using TrajoptMotionGenerator.

Robot going over a wall, while avoiding world-collisions.
The pipeline is fully Cartesian: specify start/goal SE(3) poses, and the
motion generator handles Cartesian spline interpolation, collision-free IK
seeding, and SCO trajectory optimization.
"""

import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
import trimesh
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

from pyronot.motion_generators import TrajoptMotionGenerator
from pyronot.optimization_engines import TrajOptConfig


def main():
    # Load robot
    urdf_path = "resources/panda/panda_spherized.urdf"
    mesh_dir = "resources/panda/meshes"
    ee_link_name = "panda_hand"
    down_wxyz = np.array([0, 0, 1, 0])

    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollisionSpherized.from_urdf(urdf)

    # Trajectory endpoints
    n_timesteps = 64
    start_pos = np.array([0.5, -0.3, 0.2])
    goal_pos = np.array([0.5, 0.3, 0.2])

    # Obstacles
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    wall_height = 0.4
    wall_width = 0.1
    wall_length = 0.4
    wall_intervals = np.arange(start=0.3, stop=wall_length + 0.3, step=0.05)
    wall_positions = np.column_stack([
        wall_intervals,
        np.zeros_like(wall_intervals),
        np.full_like(wall_intervals, wall_height / 2),
    ])
    wall_coll = pk.collision.Capsule.from_radius_height(
        position=wall_positions,
        radius=np.full((wall_positions.shape[0], 1), wall_width / 2),
        height=np.full((wall_positions.shape[0], 1), wall_height),
    )

    # SE(3) poses for start and goal
    def _make_se3(pos, wxyz):
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(wxyz, dtype=jnp.float32)),
            jnp.array(pos, dtype=jnp.float32),
        )

    start_pose = _make_se3(start_pos, down_wxyz)
    goal_pose = _make_se3(goal_pos, down_wxyz)

    # Plan trajectory
    motion_gen = TrajoptMotionGenerator(
        robot=robot,
        robot_coll=robot_coll,
        world_geoms=(ground_coll, wall_coll),
        ee_link_name=ee_link_name,
        n_timesteps=n_timesteps,
        cartesian_spline_mode="cubic",
        trajopt_cfg=TrajOptConfig(
            n_outer_iters=50,
            n_inner_iters=100,
            m_lbfgs=6,
            w_smooth=10.0,
            w_acc=1.0,
            w_jerk=0.5,
            w_collision=5.0,
            w_collision_max=20.0,
            penalty_scale=2.0,
            collision_margin=0.02,
            w_trust=1.0,
        ),
    )

    key = jax.random.PRNGKey(42)
    print("Running TrajoptMotionGenerator...")
    t0 = time.perf_counter()
    best_traj, costs, _, _ = motion_gen.generate(start_pose, goal_pose, key)
    best_traj.block_until_ready()
    print(f"  Done in {time.perf_counter() - t0:.2f}s  |  best cost: {float(jnp.min(costs)):.4f}")

    traj = np.array(best_traj)

    # Visualize
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(wall_length, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0.5, 0.0, wall_height / 2])
            ),
        ),
    )
    for name, pos in zip(["start", "end"], [start_pos, goal_pos]):
        server.scene.add_frame(
            f"/{name}",
            position=pos,
            wxyz=down_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )

    # Compute EE positions along the trajectory and draw the path
    ee_link_index = robot.links.names.index(ee_link_name)
    fk_all = robot.forward_kinematics(jnp.array(traj))  # (n_timesteps, n_links, 7)
    ee_positions = np.array(fk_all[:, ee_link_index, 4:7])  # xyz is last 3 of wxyz_xyz
    server.scene.add_spline_catmull_rom(
        "/trajectory_path",
        positions=ee_positions,
        color=(0, 120, 255),
        line_width=3.0,
    )

    slider = server.gui.add_slider(
        "Timestep", min=0, max=n_timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % n_timesteps
        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
