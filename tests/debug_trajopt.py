"""Visualize and compare trajectories from all solvers + cuRobo side by side.

Runs the same benchmark pipeline as bench_trajopt.py, then opens a viser
server to visualize the EE paths and animate the robot along each trajectory.

Usage
-----
    python tests/debug_trajopt.py
    python tests/debug_trajopt.py --problem bookshelf_tall --index 1
    python tests/debug_trajopt.py --disable chomp --disable stomp
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyronot._robot import Robot
from pyronot.collision._obstacles import create_collision_environment, stack_obstacles
from pyronot.collision._robot_collision import RobotCollisionSpherized
from pyronot.motion_generators import TrajoptMotionGenerator
from pyronot.optimization_engines import (
    ChompTrajOptConfig,
    LbfgsTrajOptConfig,
    LsTrajOptConfig,
    ScoTrajOptConfig,
    StompTrajOptConfig,
    chomp_trajopt,
    lbfgs_trajopt,
    ls_trajopt,
    sco_trajopt,
    stomp_trajopt,
)

# ---------------------------------------------------------------------------
# Constants (same as bench_trajopt.py)
# ---------------------------------------------------------------------------

BATCH_SIZE = 25
N_TIMESTEPS = 64
NOISE_SCALE = 0.05
EE_LINK_NAME = "panda_hand"

RESOURCES = Path(__file__).parent.parent / "resources"
PANDA_URDF = RESOURCES / "panda" / "panda_spherized.urdf"
PANDA_SRDF = RESOURCES / "panda" / "panda.srdf"
PROBLEMS_PKL = RESOURCES / "panda" / "problems.pkl"

# Colors for each solver (RGB 0-255)
# CIK = Cartesian IK seed (solid), LJS = Linear JS seed (lighter)
SOLVER_COLORS = {
    "SCO (CIK)": (0, 120, 255),
    "SCO (LJS)": (100, 180, 255),
    "LS (CIK)": (255, 120, 0),
    "LS (LJS)": (255, 180, 80),
    "CHOMP (CIK)": (0, 200, 80),
    "CHOMP (LJS)": (100, 230, 140),
    "L-BFGS (CIK)": (180, 120, 0),
    "L-BFGS (LJS)": (220, 180, 60),
    "STOMP (CIK)": (200, 0, 200),
    "STOMP (LJS)": (230, 100, 230),
    "cuRobo": (255, 40, 40),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problem(problem_name: str, index: int) -> dict:
    with open(PROBLEMS_PKL, "rb") as f:
        data = pickle.load(f)
    problems = data["problems"]
    if problem_name not in problems:
        raise ValueError(f"Problem '{problem_name}' not found. Available: {list(problems.keys())}")
    entries = problems[problem_name]
    try:
        return next(e for e in entries if e["index"] == index)
    except StopIteration:
        raise ValueError(f"No entry with index={index} in problem '{problem_name}'.")


def load_robot(urdf_path: Path):
    urdf = yourdfpy.URDF.load(urdf_path.as_posix(), mesh_dir=urdf_path.parent.as_posix())
    robot = Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=PANDA_SRDF.as_posix())
    return robot, robot_coll, urdf


def run_curobo(problem_data, start_cfg_np, goal_pose, n_timesteps):
    """Run cuRobo and return (best_traj, seed_trajs) as numpy arrays.

    Returns:
        best_traj: (T, DOF) or None if failed
        seed_trajs: list of (T, DOF) linear-interpolation seeds (start → IK goals)
    """
    jax.clear_caches()

    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    from bench_trajopt import _build_curobo_world

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
    world_cfg = _build_curobo_world(problem_data)

    num_trajopt_seeds = 4
    curobo_tsteps = min(n_timesteps, 48)
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        world_model=world_cfg if world_cfg else None,
        tensor_args=tensor_args,
        num_trajopt_seeds=num_trajopt_seeds,
        num_ik_seeds=32,
        trajopt_tsteps=curobo_tsteps,
    )
    motion_gen = MotionGen(motion_gen_cfg)
    motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)

    gp_trans = np.array(goal_pose.translation())
    gp_wxyz = np.array(goal_pose.rotation().wxyz)
    cu_goal = CuPose(
        position=torch.tensor(gp_trans, device="cuda:0", dtype=torch.float32).unsqueeze(0),
        quaternion=torch.tensor(gp_wxyz, device="cuda:0", dtype=torch.float32).unsqueeze(0),
    )
    start_js = JointState.from_position(
        torch.tensor(start_cfg_np, device="cuda:0", dtype=torch.float32).unsqueeze(0)
    )

    # --- First, run IK to get the goal configs cuRobo would use as seeds ---
    ik_result = motion_gen.solve_ik(
        cu_goal,
        retract_config=start_js.position,
        return_seeds=num_trajopt_seeds,
    )
    ik_configs = []
    if ik_result.success.any():
        # solution: (batch, return_seeds, dof), success: (batch, return_seeds)
        sol = ik_result.solution.view(-1, ik_result.solution.shape[-1])  # (N, dof)
        mask = ik_result.success.view(-1)  # (N,)
        ik_positions = sol[mask].cpu().numpy()
        ik_configs = [ik_positions[i] for i in range(min(num_trajopt_seeds, len(ik_positions)))]

    # Build linear-interpolation seed trajectories (start → each IK goal)
    seed_trajs = []
    for goal_q in ik_configs:
        alphas = np.linspace(0.0, 1.0, curobo_tsteps).reshape(-1, 1)
        seed = start_cfg_np * (1.0 - alphas) + goal_q * alphas  # (T, DOF)
        seed_trajs.append(seed)

    # --- Now run full planning ---
    plan_cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = motion_gen.plan_single(start_js, cu_goal, plan_cfg)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if result.success.item():
        traj_np = result.optimized_plan.position.cpu().numpy()  # (T, DOF)
        print(f"  cuRobo: solved in {elapsed:.3f}s, {traj_np.shape[0]} timesteps, {len(seed_trajs)} IK seeds")
        return traj_np, seed_trajs
    else:
        print(f"  cuRobo: FAILED ({elapsed:.3f}s), {len(seed_trajs)} IK seeds")
        return None, seed_trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(problem_name: str, index: int, disabled_solvers: set[str]) -> None:
    print(f"Loading problem '{problem_name}' index={index}...")
    problem_data = load_problem(problem_name, index)
    start_cfg = jnp.array(problem_data["start"], dtype=jnp.float32)
    goal_cfg = jnp.array(problem_data["goals"], dtype=jnp.float32)[0]

    robot, robot_coll, urdf = load_robot(PANDA_URDF)
    ee_idx = robot.links.names.index(EE_LINK_NAME)

    start_pose = jaxlie.SE3(robot.forward_kinematics(start_cfg)[ee_idx])
    goal_pose = jaxlie.SE3(robot.forward_kinematics(goal_cfg)[ee_idx])

    mid_pos = (start_pose.translation() + goal_pose.translation()) / 2.0
    mid_rot = start_pose.rotation() @ jaxlie.SO3.exp(
        0.5 * (start_pose.rotation().inverse() @ goal_pose.rotation()).log()
    )
    mid_pose = jaxlie.SE3.from_rotation_and_translation(mid_rot, mid_pos)

    obstacles = create_collision_environment(problem_data)
    world_geoms = stack_obstacles(obstacles)

    control_wxyz_xyz = jnp.concatenate(
        [start_pose.wxyz_xyz[None], mid_pose.wxyz_xyz[None], goal_pose.wxyz_xyz[None]],
        axis=0,
    )
    control_poses = jaxlie.SE3(control_wxyz_xyz)

    # --- Solver configs ---
    sco_cfg = ScoTrajOptConfig(
        n_outer_iters=10, n_inner_iters=30, m_lbfgs=6,
        w_smooth=1.0, w_vel=1.0, w_acc=0.5, w_jerk=0.1,
        w_collision=10.0, w_collision_max=100.0, penalty_scale=3.0,
        collision_margin=0.02, w_trust=0.5, w_limits=1.0,
    )
    ls_cfg = LsTrajOptConfig(
        n_outer_iters=10, n_ls_iters=10, lambda_init=5e-3,
        w_smooth=1.0, w_acc=0.5, w_jerk=0.1,
        w_collision=10.0, w_collision_max=100.0, penalty_scale=3.0,
        collision_margin=0.02, w_trust=0.5, w_limits=1.0,
        w_endpoint=100.0, smooth_min_temperature=0.05, max_delta_per_step=0.1,
    )
    chomp_cfg = ChompTrajOptConfig(
        n_iters=100, step_size=0.2,
        w_smooth=1.0, w_vel=1.0, w_acc=0.5, w_jerk=0.1,
        w_collision=10.0, collision_margin=0.02, w_limits=1.0,
        use_covariant_update=True, smoothness_reg=1e-3,
    )
    lbfgs_cfg = LbfgsTrajOptConfig(
        n_iters=100, m_lbfgs=6,
        w_smooth=1.0, w_vel=1.0, w_acc=0.5, w_jerk=0.1,
        w_collision=10.0, w_collision_max=100.0, penalty_scale=2.0,
        escalation_interval=20, collision_margin=0.02, w_limits=1.0,
    )
    stomp_cfg = StompTrajOptConfig(
        n_iters=40, n_samples=96, noise_scale=0.03,
        temperature=1.0, step_size=0.3,
        w_smooth=1.0, w_vel=1.0, w_acc=0.5, w_jerk=0.1,
        w_collision=10.0, w_collision_max=100.0,
        collision_penalty_scale=1.05, collision_margin=0.02, w_limits=1.0,
        use_covariant_update=False, smoothness_reg=0.1,
        use_cost_normalization=True, use_null_particle=True,
        use_elite_filter=True, elite_frac=0.25,
        adaptive_covariance=True, cov_update_rate=0.2,
        noise_decay=0.99, noise_scale_min=0.003, noise_scale_max=0.1,
        normalize_smooth_noise_scale=True,
        n_lbfgs_iters=10, m_lbfgs=5, lbfgs_step_scale=1.0,
    )

    # --- Seed trajectories with both modes ---
    key = jax.random.PRNGKey(0)

    seed_modes = [
        ("cartesian_ik", "CIK"),   # Cartesian IK seeding (original)
        ("linear_js", "LJS"),       # Linear joint-space seeding (cuRobo-style)
    ]

    # Store per-mode: seeds (B, T, DOF), init_trajs, start, goal
    seeded: dict[str, dict] = {}
    for smode, label in seed_modes:
        motion_gen = TrajoptMotionGenerator(
            robot=robot, robot_coll=robot_coll, world_geoms=world_geoms,
            ee_link_name=EE_LINK_NAME, n_timesteps=N_TIMESTEPS,
            n_batch=BATCH_SIZE, noise_scale=NOISE_SCALE,
            cartesian_spline_mode="bspline", trajopt_cfg=sco_cfg, use_cuda=False,
            seed_mode=smode,
        )
        # For linear_js, pass ground-truth start/goal configs from the problem
        # file to skip IK — same as what cuRobo gets.
        extra_seed_kwargs = {}
        if smode == "linear_js":
            extra_seed_kwargs = {"start_cfg": start_cfg, "goal_cfg": goal_cfg}

        key, k_wu = jax.random.split(key)
        motion_gen._seed_trajectories(control_poses, k_wu, **extra_seed_kwargs)  # warmup

        key, k_seed = jax.random.split(key)
        init_trajs, seeded_start, seeded_goal = motion_gen._seed_trajectories(
            control_poses, k_seed, **extra_seed_kwargs
        )
        init_trajs.block_until_ready()
        print(f"Seeded [{label}]: smoothness={float(jnp.mean(jnp.sum((init_trajs[0,1:] - init_trajs[0,:-1])**2, axis=-1))):.4f}")
        seeded[label] = {
            "init_trajs": init_trajs,
            "start": seeded_start,
            "goal": seeded_goal,
            "seeds_np": np.array(init_trajs),
        }

    # --- Run solvers on both seed modes ---
    solver_runs = []
    if "sco" not in disabled_solvers:
        solver_runs.append(("SCO", sco_trajopt, sco_cfg, {}))
    if "ls" not in disabled_solvers:
        solver_runs.append(("LS", ls_trajopt, ls_cfg, {"key": jax.random.PRNGKey(123)}))
    if "lbfgs" not in disabled_solvers:
        solver_runs.append(("L-BFGS", lbfgs_trajopt, lbfgs_cfg, {}))
    if "chomp" not in disabled_solvers:
        solver_runs.append(("CHOMP", chomp_trajopt, chomp_cfg, {}))
    if "stomp" not in disabled_solvers:
        solver_runs.append(("STOMP", stomp_trajopt, stomp_cfg, {"key": jax.random.PRNGKey(42)}))

    # Dict of display_name -> (T, DOF) numpy trajectory
    trajectories: dict[str, np.ndarray] = {}

    for seed_label, seed_data in seeded.items():
        it = seed_data["init_trajs"]
        ss = seed_data["start"]
        sg = seed_data["goal"]
        for name, solver_fn, solver_cfg, extra in solver_runs:
            display = f"{name} ({seed_label})"
            print(f"Running {display}...")
            try:
                # Warmup
                solver_fn(
                    it, ss, sg, robot, robot_coll, world_geoms, solver_cfg,
                    use_cuda=False, **extra,
                )
                # Timed
                t0 = time.perf_counter()
                best_traj, costs, final_trajs = solver_fn(
                    it, ss, sg, robot, robot_coll, world_geoms, solver_cfg,
                    use_cuda=False, **extra,
                )
                best_traj.block_until_ready()
                elapsed = time.perf_counter() - t0
                trajectories[display] = np.array(best_traj)
                print(f"  {display}: {elapsed:.3f}s, best cost: {float(jnp.min(costs)):.4f}")
            except Exception as exc:
                print(f"  {display}: FAILED — {exc}")

    # --- cuRobo ---
    curobo_seed_trajs: list[np.ndarray] = []
    if "curobo" not in disabled_solvers:
        print("Running cuRobo...")
        try:
            curobo_traj, curobo_seed_trajs = run_curobo(
                problem_data, np.array(start_cfg), goal_pose, N_TIMESTEPS
            )
            if curobo_traj is not None:
                trajectories["cuRobo"] = curobo_traj
        except Exception as exc:
            print(f"  cuRobo: FAILED — {exc}")

    if not trajectories:
        print("No trajectories to visualize!")
        return

    # ===================================================================
    # Visualization
    # ===================================================================
    print(f"\nVisualizing {len(trajectories)} trajectories: {list(trajectories.keys())}")
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # Robot
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(np.array(start_cfg))

    # Obstacles
    for i, obs in enumerate(obstacles):
        if hasattr(obs, "to_trimesh"):
            mesh = obs.to_trimesh()
            server.scene.add_mesh_trimesh(f"/obstacles/obj_{i}", mesh=mesh)

    # Start / goal frames
    for label, pose in [("start", start_pose), ("goal", goal_pose)]:
        server.scene.add_frame(
            f"/{label}",
            position=np.array(pose.translation()),
            wxyz=np.array(pose.rotation().wxyz),
            axes_length=0.05,
            axes_radius=0.01,
        )

    # --- Add seed trajectories to the trajectories dict ---
    for seed_label, seed_data in seeded.items():
        seeds_np = seed_data["seeds_np"]
        for b in range(seeds_np.shape[0]):
            trajectories[f"{seed_label}_Seed_{b}"] = seeds_np[b]

    for i, st in enumerate(curobo_seed_trajs):
        trajectories[f"cuRobo_Seed_{i}"] = st

    # --- GUI controls ---
    solver_names = list(trajectories.keys())
    solver_dropdown = server.gui.add_dropdown(
        "Solver", options=solver_names, initial_value=solver_names[0]
    )
    max_T = max(t.shape[0] for t in trajectories.values())
    slider = server.gui.add_slider("Timestep", min=0, max=max_T - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Playing", initial_value=True)
    speed = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)

    # Draw EE paths and store handles for visibility toggling
    path_handles: dict[str, object] = {}
    vis_checkboxes: dict[str, object] = {}

    # Seed color map: group label -> (color, line_width)
    SEED_STYLE = {
        "CIK": ((160, 160, 160), 1.0),      # Cartesian IK seeds — gray
        "LJS": ((160, 220, 160), 1.0),      # Linear JS seeds — light green
        "cuRobo": ((255, 160, 160), 1.0),   # cuRobo seeds — light red
    }

    def _seed_group(name: str) -> str | None:
        """Return the seed group label if name is a seed, else None."""
        for prefix in ("CIK_Seed_", "LJS_Seed_", "cuRobo_Seed_"):
            if name.startswith(prefix):
                return prefix.split("_Seed_")[0]
        return None

    # Draw paths and build visibility toggles
    with server.gui.add_folder("Path Visibility"):
        for name, traj in trajectories.items():
            group = _seed_group(name)
            fk_all = robot.forward_kinematics(jnp.array(traj))
            ee_positions = np.array(fk_all[:, ee_idx, 4:7])

            if group is not None:
                color, line_width = SEED_STYLE[group]
            else:
                color = SOLVER_COLORS.get(name, (180, 180, 180))
                line_width = 3.0

            path_handles[name] = server.scene.add_spline_catmull_rom(
                f"/paths/{name}",
                positions=ee_positions,
                color=color,
                line_width=line_width,
            )

        # Per-solver toggles
        for name in trajectories:
            if _seed_group(name) is None:
                vis_checkboxes[name] = server.gui.add_checkbox(name, initial_value=True)
        # Grouped seed toggles
        vis_checkboxes["CIK Seeds"] = server.gui.add_checkbox("CIK Seeds (Cartesian IK)", initial_value=True)
        vis_checkboxes["LJS Seeds"] = server.gui.add_checkbox("LJS Seeds (Linear JS)", initial_value=True)
        vis_checkboxes["cuRobo Seeds"] = server.gui.add_checkbox("cuRobo Seeds", initial_value=True)

    print("Viser running — open the URL above. Press Ctrl+C to stop.")
    try:
        while True:
            # Update path visibility
            seed_vis = {
                "CIK": vis_checkboxes["CIK Seeds"].value,
                "LJS": vis_checkboxes["LJS Seeds"].value,
                "cuRobo": vis_checkboxes["cuRobo Seeds"].value,
            }
            for name, handle in path_handles.items():
                group = _seed_group(name)
                if group is not None:
                    handle.visible = seed_vis[group]
                elif name in vis_checkboxes:
                    handle.visible = vis_checkboxes[name].value

            # Advance timestep
            if playing.value:
                slider.value = (slider.value + 1) % max_T

            # Get active trajectory and update robot
            active_name = solver_dropdown.value
            traj = trajectories[active_name]
            t = min(slider.value, traj.shape[0] - 1)
            urdf_vis.update_cfg(traj[t])

            time.sleep(1.0 / speed.value)
    except KeyboardInterrupt:
        print("Stopping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajopt solutions with viser.")
    parser.add_argument("--problem", default="bookshelf_thin")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument(
        "--disable", action="append", default=[],
        choices=["sco", "ls", "lbfgs", "chomp", "stomp", "curobo"],
    )
    args = parser.parse_args()
    main(args.problem, args.index, set(args.disable))
