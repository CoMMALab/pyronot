"""Benchmark: SCO TrajOpt on a VAMP problem instance.

Loads a planning problem from the VAMP dataset (panda/problems.pkl), runs FK
to obtain Cartesian start/goal poses, then benchmarks ``TrajoptMotionGenerator``
with each Cartesian spline mode (linear, cubic, bspline).

Usage
-----
    python tests/bench_trajopt.py [--problem bookshelf_tall] [--index 1]

The script prints a results table with columns:
    Method | Warmup (s) | Run (s) | TrajOpt (s) | Smoothness | Solved (B=25)

"Solved" counts trajectories whose final waypoint is within ``GOAL_TOL`` rad of
the goal configuration AND whose minimum world/self-collision distance is above
``COLL_TOL`` for every timestep.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import yourdfpy

# ---------------------------------------------------------------------------
# Path setup so the script works from the repo root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyronot._robot import Robot
from pyronot.collision._obstacles import create_collision_environment, stack_obstacles
from pyronot.collision._robot_collision import RobotCollisionSpherized
from pyronot.motion_generators import TrajoptMotionGenerator
from pyronot.optimization_engines._sco_optimization import TrajOptConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE   = 25
N_TIMESTEPS  = 64
NOISE_SCALE  = 0.05   # std of per-seed perturbation
EE_LINK_NAME = "panda_hand"
GOAL_POS_TOL = 0.01   # m — EE position must be within this of goal pose
GOAL_ORI_TOL = 0.05   # rad — EE orientation must be within this of goal pose
COLL_TOL     = 0.0    # signed-distance threshold to count as collision-free

RESOURCES    = Path(__file__).parent.parent / "resources"
PANDA_URDF   = RESOURCES / "panda" / "panda_spherized.urdf"
PANDA_SRDF   = RESOURCES / "panda" / "panda.srdf"
PROBLEMS_PKL = RESOURCES / "panda" / "problems.pkl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problem(problem_name: str, index: int) -> dict:
    with open(PROBLEMS_PKL, "rb") as f:
        data = pickle.load(f)
    problems = data["problems"]
    if problem_name not in problems:
        available = list(problems.keys())
        raise ValueError(f"Problem '{problem_name}' not found. Available: {available}")
    entries = problems[problem_name]
    try:
        return next(e for e in entries if e["index"] == index)
    except StopIteration:
        raise ValueError(f"No entry with index={index} in problem '{problem_name}'.")


def load_robot(urdf_path: Path) -> tuple[Robot, RobotCollisionSpherized, yourdfpy.URDF]:
    urdf = yourdfpy.URDF.load(
        urdf_path.as_posix(),
        mesh_dir=urdf_path.parent.as_posix(),
    )
    robot      = Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=PANDA_SRDF.as_posix())
    return robot, robot_coll, urdf


def compute_smoothness(traj: jnp.ndarray) -> float:
    """Mean squared velocity norm over the trajectory."""
    vel = traj[1:] - traj[:-1]           # [T-1, DOF]
    return float(jnp.mean(jnp.sum(vel ** 2, axis=-1)))


def check_solved(
    trajs: jnp.ndarray,           # [B, T, DOF]
    goal_pose: jaxlie.SE3,
    ee_idx: int,
    robot: Robot,
    robot_coll: RobotCollisionSpherized,
    world_geoms: list,
) -> int:
    """Count trajectories that reach the goal pose without collisions.

    Goal-reaching is evaluated in Cartesian space: the EE pose at the final
    waypoint is compared to ``goal_pose`` in both position and orientation.
    """
    n_solved = 0
    for b in range(trajs.shape[0]):
        traj = trajs[b]                   # [T, DOF]

        # Goal condition: EE pose at final waypoint close to goal pose
        final_pose = jaxlie.SE3(robot.forward_kinematics(traj[-1])[ee_idx])
        pos_err = float(jnp.linalg.norm(final_pose.translation() - goal_pose.translation()))
        ori_err = float(jnp.linalg.norm((final_pose.rotation().inverse() @ goal_pose.rotation()).log()))
        if pos_err > GOAL_POS_TOL or ori_err > GOAL_ORI_TOL:
            continue

        # Collision condition: check every timestep
        collision_free = True
        for t in range(traj.shape[0]):
            cfg = traj[t]

            # Self-collision
            self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
            if float(jnp.min(self_dists)) < COLL_TOL:
                collision_free = False
                break

            # World collision — one geometry type at a time
            for world_geom in world_geoms:
                world_dists = robot_coll.compute_world_collision_distance(
                    robot, cfg, world_geom
                )
                if float(jnp.min(world_dists)) < COLL_TOL:
                    collision_free = False
                    break
            if not collision_free:
                break

        if collision_free:
            n_solved += 1

    return n_solved


def _fmt_row(name: str, warmup: float, elapsed: float, trajopt_time: float, smoothness: float, solved: int) -> str:
    return f"  {name:<36s}  {warmup:>10.3f}  {elapsed:>10.3f}  {trajopt_time:>12.3f}  {smoothness:>12.4f}  {solved:>6d}/{BATCH_SIZE}"


def _run_motion_generator(
    name: str,
    motion_gen: TrajoptMotionGenerator,
    start_pose: jaxlie.SE3,
    goal_pose: jaxlie.SE3,
    key: jnp.ndarray,
    waypoint_poses: jaxlie.SE3 | None = None,
) -> dict:
    """Run warm-up + timed TrajoptMotionGenerator and return metrics dict."""
    # Warm-up / JIT compile
    key, wu_key = jax.random.split(key)
    t0_wu = time.perf_counter()
    best_wu, _, _, _ = motion_gen.generate(
        start_pose, goal_pose, wu_key, waypoint_poses=waypoint_poses,
    )
    best_wu.block_until_ready()
    warmup_elapsed = time.perf_counter() - t0_wu

    # Timed run
    key, run_key = jax.random.split(key)
    t0 = time.perf_counter()
    best_traj, costs, final_trajs, trajopt_time = motion_gen.generate(
        start_pose, goal_pose, run_key, waypoint_poses=waypoint_poses,
    )
    best_traj.block_until_ready()
    elapsed = time.perf_counter() - t0

    ee_idx = motion_gen.robot.links.names.index(motion_gen.ee_link_name)
    smoothness = float(jnp.mean(
        jnp.array([compute_smoothness(final_trajs[b]) for b in range(BATCH_SIZE)])
    ))
    n_solved = check_solved(
        final_trajs, goal_pose, ee_idx, motion_gen.robot, motion_gen.robot_coll,
        list(motion_gen.world_geoms),
    )
    best_cost = float(jnp.min(costs))

    return {
        "name": name,
        "warmup": warmup_elapsed,
        "elapsed": elapsed,
        "trajopt_time": trajopt_time,
        "smoothness": smoothness,
        "n_solved": n_solved,
        "best_cost": best_cost,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main(problem_name: str, index: int) -> None:
    print(f"\n=== SCO TrajOpt Benchmark  |  problem={problem_name!r}  index={index} ===\n")

    # --- Load problem data ---------------------------------------------------
    print("Loading problem...")
    problem_data = load_problem(problem_name, index)
    start_cfg = jnp.array(problem_data["start"], dtype=jnp.float32)
    goals = jnp.array(problem_data["goals"], dtype=jnp.float32)
    goal_cfg  = goals[0]
    valid = problem_data.get("valid", True)
    print(f"  Start cfg : {np.array(start_cfg)}")
    print(f"  Goal  cfg : {np.array(goal_cfg)}")
    print(f"  Valid     : {valid}")

    # --- Load robot ----------------------------------------------------------
    print("\nLoading robot...")
    robot, robot_coll, urdf = load_robot(PANDA_URDF)
    dof = robot.joints.num_actuated_joints
    ee_idx = robot.links.names.index(EE_LINK_NAME)
    print(f"  DOF   : {dof}")

    # --- FK to get Cartesian start/goal poses ---------------------------------
    print("\nRunning FK for start/goal Cartesian poses...")
    start_pose = jaxlie.SE3(robot.forward_kinematics(start_cfg)[ee_idx])
    goal_pose  = jaxlie.SE3(robot.forward_kinematics(goal_cfg)[ee_idx])
    print(f"  Start pos : {np.array(start_pose.translation())}")
    print(f"  Goal  pos : {np.array(goal_pose.translation())}")

    # Cartesian midpoint for K=3 spline control poses
    mid_pos = (start_pose.translation() + goal_pose.translation()) / 2.0
    mid_rot = start_pose.rotation() @ jaxlie.SO3.exp(
        0.5 * (start_pose.rotation().inverse() @ goal_pose.rotation()).log()
    )
    mid_pose = jaxlie.SE3.from_rotation_and_translation(mid_rot, mid_pos)

    # --- Build collision environment ----------------------------------------
    print("\nBuilding collision environment...")
    obstacles = create_collision_environment(problem_data)
    world_geoms = stack_obstacles(obstacles)      # tuple of per-type stacked geoms
    print(f"  Geometry groups : {len(world_geoms)}")
    for wg in world_geoms:
        print(f"    {type(wg).__name__}  batch={wg.get_batch_axes()}")

    # --- TrajOpt configuration -----------------------------------------------
    opt_cfg = TrajOptConfig(
        n_outer_iters=10,
        n_inner_iters=30,
        m_lbfgs=6,
        w_smooth=1.0,
        w_vel=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        w_collision_max=100.0,
        penalty_scale=3.0,
        collision_margin=0.02,
        w_trust=0.5,
        w_limits=1.0,
    )

    key = jax.random.PRNGKey(0)
    results = []

    # --- Benchmark each Cartesian spline mode --------------------------------
    for mode in ("linear", "cubic", "bspline"):
        # For linear mode, K=2 is fine. For cubic/bspline, use K=3 with midpoint.
        if mode == "linear":
            wp = None
            label_suffix = "(K=2)"
        else:
            wp = mid_pose
            label_suffix = "(K=3)"

        motion_gen = TrajoptMotionGenerator(
            robot=robot,
            robot_coll=robot_coll,
            world_geoms=world_geoms,
            ee_link_name=EE_LINK_NAME,
            n_timesteps=N_TIMESTEPS,
            n_batch=BATCH_SIZE,
            noise_scale=NOISE_SCALE,
            cartesian_spline_mode=mode,
            trajopt_cfg=opt_cfg,
            use_cuda=False,
        )

        key, subkey = jax.random.split(key)
        print(f"\n[JAX] {mode} {label_suffix} — warm-up + timed run...")
        res = _run_motion_generator(
            f"JAX  {mode:>8s} {label_suffix}",
            motion_gen, start_pose, goal_pose, subkey,
            waypoint_poses=wp,
        )
        results.append(res)

        try:
            motion_gen_cuda = TrajoptMotionGenerator(
                robot=robot,
                robot_coll=robot_coll,
                world_geoms=world_geoms,
                ee_link_name=EE_LINK_NAME,
                n_timesteps=N_TIMESTEPS,
                n_batch=BATCH_SIZE,
                noise_scale=NOISE_SCALE,
                cartesian_spline_mode=mode,
                trajopt_cfg=opt_cfg,
                use_cuda=True,
            )
            key, subkey = jax.random.split(key)
            print(f"[CUDA] {mode} {label_suffix} — warm-up + timed run...")
            res_cu = _run_motion_generator(
                f"CUDA {mode:>8s} {label_suffix}",
                motion_gen_cuda, start_pose, goal_pose, subkey,
                waypoint_poses=wp,
            )
            results.append(res_cu)
        except Exception as exc:
            print(f"  CUDA benchmark skipped: {exc}")

    # --- Results table -------------------------------------------------------
    print("\n" + "=" * 80)
    header = (
        f"  {'Method':<36s}  {'Warmup (s)':>10s}  {'Run (s)':>10s}  {'TrajOpt (s)':>12s}  {'Smoothness':>12s}  {'Solved':>8s}\n"
        f"  {'-'*36}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}"
    )
    print(header)
    for r in results:
        print(_fmt_row(r["name"], r["warmup"], r["elapsed"], r["trajopt_time"], r["smoothness"], r["n_solved"]))

    print()
    for r in results:
        print(f"  {r['name']:<36s}  best_cost = {r['best_cost']:.4f}")

    print(f"\n  Batch size (B)                 : {BATCH_SIZE}")
    print(f"  Timesteps (T)                  : {N_TIMESTEPS}")
    print(f"  DOF                            : {dof}")
    print(f"  Outer iterations               : {opt_cfg.n_outer_iters}")
    print(f"  Inner L-BFGS iters             : {opt_cfg.n_inner_iters}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SCO TrajOpt on a VAMP problem.")
    parser.add_argument("--problem", default="bookshelf_tall", help="Problem name in problems.pkl")
    parser.add_argument("--index",   type=int, default=1,     help="Problem instance index")
    args = parser.parse_args()
    main(args.problem, args.index)
