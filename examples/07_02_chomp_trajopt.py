"""Trajectory Optimization with CHOMP.

Robot goes over a wall while avoiding world collisions.
This example reuses the Cartesian seeding pipeline and runs CHOMP as the
trajectory optimizer.
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
from pyronot.optimization_engines import ChompTrajOptConfig, ScoTrajOptConfig, chomp_trajopt


def main(
    use_cuda: bool = False,
    n_iters: int = 220,
    step_size: float = 0.12,
    w_smooth: float = 2.0,
    w_acc: float = 0.8,
    w_jerk: float = 0.25,
    smoothness_reg: float = 1e-3,
    grad_clip_norm: float = 10.0,
    max_delta_per_step: float = 0.05,
):
    urdf_path = "resources/panda/panda_spherized.urdf"
    mesh_dir = "resources/panda/meshes"
    ee_link_name = "panda_hand"
    down_wxyz = np.array([0, 0, 1, 0])

    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollisionSpherized.from_urdf(urdf)

    n_timesteps = 64
    n_batch = 25
    start_pos = np.array([0.5, -0.3, 0.2])
    goal_pos = np.array([0.5, 0.3, 0.2])

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

    def _make_se3(pos, wxyz):
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(wxyz, dtype=jnp.float32)),
            jnp.array(pos, dtype=jnp.float32),
        )

    start_pose = _make_se3(start_pos, down_wxyz)
    goal_pose = _make_se3(goal_pos, down_wxyz)

    # Reuse TrajoptMotionGenerator only for Cartesian spline + IK seeding.
    seed_gen = TrajoptMotionGenerator(
        robot=robot,
        robot_coll=robot_coll,
        world_geoms=(ground_coll, wall_coll),
        ee_link_name=ee_link_name,
        n_timesteps=n_timesteps,
        n_batch=n_batch,
        noise_scale=0.05,
        cartesian_spline_mode="cubic",
        trajopt_cfg=ScoTrajOptConfig(),
    )

    control_poses = jaxlie.SE3(jnp.concatenate([
        start_pose.wxyz_xyz[None],
        goal_pose.wxyz_xyz[None],
    ], axis=0))

    key = jax.random.PRNGKey(42)
    init_trajs, start_cfg, goal_cfg = seed_gen._seed_trajectories(control_poses, key)

    print("Running CHOMP TrajOpt...")
    t0 = time.perf_counter()
    best_traj, costs, _ = chomp_trajopt(
        init_trajs,
        start_cfg,
        goal_cfg,
        robot,
        robot_coll,
        (ground_coll, wall_coll),
        ChompTrajOptConfig(
            n_iters=n_iters,
            step_size=step_size,
            w_smooth=w_smooth,
            w_acc=w_acc,
            w_jerk=w_jerk,
            w_collision=10.0,
            collision_margin=0.02,
            w_limits=1.0,
            use_covariant_update=True,
            smoothness_reg=smoothness_reg,
            grad_clip_norm=grad_clip_norm,
            max_delta_per_step=max_delta_per_step,
        ),
        use_cuda=use_cuda,
    )
    best_traj.block_until_ready()
    print(f"  Done in {time.perf_counter() - t0:.2f}s  |  best cost: {float(jnp.min(costs)):.4f}")

    traj = np.array(best_traj)

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

    ee_link_index = robot.links.names.index(ee_link_name)
    fk_all = robot.forward_kinematics(jnp.array(traj))
    ee_positions = np.array(fk_all[:, ee_link_index, 4:7])
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
    print("Viewer running at http://localhost:8080  |  Press Ctrl+C to exit.")

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % n_timesteps
        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
