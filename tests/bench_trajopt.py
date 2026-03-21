"""Benchmark: SCO TrajOpt on a VAMP problem instance.

Loads a planning problem from the VAMP dataset (panda/problems.pkl), builds a
batch of B=25 B-spline-initialised trajectories with T=64 waypoints, runs
``sco_trajopt`` and reports timing, smoothness, and solve-rate.

Usage
-----
    python tests/bench_trajopt.py [--problem bookshelf_tall] [--index 1]

The script prints a results table with columns:
    Method | Time (s) | Smoothness | Solved (B=25)

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
import numpy as np
import yourdfpy

# ---------------------------------------------------------------------------
# Path setup so the script works from the repo root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pyronot as pk
from pyronot._robot import Robot
from pyronot._splines import make_spline_init_trajs
from pyronot.collision._obstacles import create_collision_environment, stack_obstacles
from pyronot.collision._robot_collision import RobotCollisionSpherized
from pyronot.optimization_engines._sco_optimization import TrajOptConfig, sco_trajopt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE   = 25
N_TIMESTEPS  = 64
NOISE_SCALE  = 0.05   # std of per-seed perturbation
GOAL_TOL     = 0.05   # rad — final waypoint must be within this of goal
COLL_TOL     = 0.0    # signed-distance threshold to count as collision-free

RESOURCES    = Path(__file__).parent.parent / "resources"
PANDA_URDF   = RESOURCES / "panda" / "panda_spherized.urdf"
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
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)
    return robot, robot_coll, urdf


def compute_smoothness(traj: jnp.ndarray) -> float:
    """Mean squared velocity norm over the trajectory."""
    vel = traj[1:] - traj[:-1]           # [T-1, DOF]
    return float(jnp.mean(jnp.sum(vel ** 2, axis=-1)))


def check_solved(
    trajs: jnp.ndarray,           # [B, T, DOF]
    goal: jnp.ndarray,            # [DOF]
    robot: Robot,
    robot_coll: RobotCollisionSpherized,
    world_geoms: list,
) -> int:
    """Count trajectories that reach the goal without collisions."""
    n_solved = 0
    for b in range(trajs.shape[0]):
        traj = trajs[b]                   # [T, DOF]

        # Goal condition: final waypoint close to goal
        goal_err = float(jnp.linalg.norm(traj[-1] - goal))
        if goal_err > GOAL_TOL:
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


def _fmt_row(name: str, elapsed: float, smoothness: float, solved: int) -> str:
    return f"  {name:<22s}  {elapsed:>10.3f}  {smoothness:>12.4f}  {solved:>6d}/{BATCH_SIZE}"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main(problem_name: str, index: int) -> None:
    print(f"\n=== SCO TrajOpt Benchmark  |  problem={problem_name!r}  index={index} ===\n")

    # --- Load problem data ---------------------------------------------------
    print("Loading problem...")
    problem_data = load_problem(problem_name, index)
    start = jnp.array(problem_data["start"], dtype=jnp.float32)
    goals = jnp.array(problem_data["goals"], dtype=jnp.float32)
    goal  = goals[0]                     # use the first goal configuration
    valid = problem_data.get("valid", True)
    print(f"  Start : {np.array(start)}")
    print(f"  Goal  : {np.array(goal)}")
    print(f"  Valid : {valid}")

    # --- Load robot ----------------------------------------------------------
    print("\nLoading robot...")
    robot, robot_coll, urdf = load_robot(PANDA_URDF)
    dof = robot.joints.num_actuated_joints
    print(f"  DOF   : {dof}")

    # --- Build collision environment ----------------------------------------
    print("\nBuilding collision environment...")
    obstacles = create_collision_environment(problem_data)
    world_geoms = stack_obstacles(obstacles)      # tuple of per-type stacked geoms
    print(f"  Geometry groups : {len(world_geoms)}")
    for wg in world_geoms:
        print(f"    {type(wg).__name__}  batch={wg.get_batch_axes()}")

    # For sco_trajopt we pass only one geometry at a time (per type). Wrap to
    # a list so the cost function can iterate; we benchmark with the first
    # non-empty group to keep the JIT-static shape requirement satisfied.
    # Full multi-type collision is handled in check_solved via Python loops.
    primary_world_geom = world_geoms[0] if world_geoms else None

    # --- Build B-spline initial trajectories --------------------------------
    print(f"\nBuilding B-spline initial trajectories (B={BATCH_SIZE}, T={N_TIMESTEPS})...")
    key = jax.random.PRNGKey(0)

    # Two control points: straight line from start to goal
    control_points = jnp.stack([start, goal], axis=0)   # [2, DOF]

    init_trajs = make_spline_init_trajs(
        control_points=control_points,
        n_batch=BATCH_SIZE,
        n_points=N_TIMESTEPS,
        key=key,
        mode="bspline",
        noise_scale=NOISE_SCALE,
        bspline_degree=1,                # degree-1 B-spline with 2 CPs = linear interpolation
    )                                    # [B, T, DOF]
    print(f"  init_trajs shape : {init_trajs.shape}")

    # --- TrajOpt configuration -----------------------------------------------
    opt_cfg = TrajOptConfig(
        n_iters=100,
        lr=5e-3,
        use_line_search=True,
        w_smooth=1.0,
        w_vel=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        collision_margin=0.02,
        w_goal=5.0,
        w_limits=1.0,
    )

    # --- Warm-up JIT compile -------------------------------------------------
    print("\nWarm-up / JIT compilation (first call)...")
    t0 = time.perf_counter()
    best_traj_warmup, costs_warmup, final_trajs_warmup = sco_trajopt(
        init_trajs, goal, robot, robot_coll, primary_world_geom, opt_cfg
    )
    best_traj_warmup.block_until_ready()
    compile_time = time.perf_counter() - t0
    print(f"  Compile + first run : {compile_time:.3f} s")

    # --- Timed benchmark run -------------------------------------------------
    print("\nTimed benchmark run (second call)...")
    t0 = time.perf_counter()
    best_traj, costs, final_trajs = sco_trajopt(
        init_trajs, goal, robot, robot_coll, primary_world_geom, opt_cfg
    )
    best_traj.block_until_ready()
    elapsed = time.perf_counter() - t0

    # --- Metrics -------------------------------------------------------------
    print("\nComputing metrics...")
    smoothness_all = float(jnp.mean(
        jnp.array([compute_smoothness(final_trajs[b]) for b in range(BATCH_SIZE)])
    ))
    smoothness_best = compute_smoothness(best_traj)

    n_solved = check_solved(final_trajs, goal, robot, robot_coll, list(world_geoms))
    best_cost = float(jnp.min(costs))

    # --- Results table -------------------------------------------------------
    header = (
        "\n"
        f"  {'Method':<22s}  {'Time (s)':>10s}  {'Smoothness':>12s}  {'Solved':>8s}\n"
        f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*8}"
    )
    print(header)
    print(_fmt_row("SCO-TrajOpt (JIT run)", elapsed, smoothness_all, n_solved))
    print(f"\n  Best trajectory cost      : {best_cost:.4f}")
    print(f"  Best trajectory smoothness: {smoothness_best:.4f}")
    print(f"  Compile + first-run time  : {compile_time:.3f} s")
    print(f"  Batch size (B)            : {BATCH_SIZE}")
    print(f"  Timesteps (T)             : {N_TIMESTEPS}")
    print(f"  DOF                       : {dof}")
    print(f"  Iterations                : {opt_cfg.n_iters}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SCO TrajOpt on a VAMP problem.")
    parser.add_argument("--problem", default="bookshelf_tall", help="Problem name in problems.pkl")
    parser.add_argument("--index",   type=int, default=1,     help="Problem instance index")
    args = parser.parse_args()
    main(args.problem, args.index)
