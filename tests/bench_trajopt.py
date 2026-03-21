"""Benchmark: SCO TrajOpt on a VAMP problem instance.

Loads a planning problem from the VAMP dataset (panda/problems.pkl), builds a
batch of B=25 Cartesian-IK-initialised trajectories with T=64 waypoints, runs
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
import jaxlie
import numpy as np
import yourdfpy

# ---------------------------------------------------------------------------
# Path setup so the script works from the repo root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pyronot as pk
from pyronot._robot import Robot
from pyronot.collision import colldist_from_sdf
from pyronot.collision._obstacles import create_collision_environment, stack_obstacles
from pyronot.collision._robot_collision import RobotCollisionSpherized
from pyronot.optimization_engines._mppi_ik import mppi_ik_solve_cuda_batch
from pyronot.optimization_engines._sco_optimization import TrajOptConfig, sco_trajopt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE   = 25
N_TIMESTEPS  = 64
NOISE_SCALE  = 0.05   # std of per-seed perturbation
EE_LINK_NAME = "panda_hand"
GOAL_TOL     = 0.05   # rad — final waypoint must be within this of goal
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


def make_cartesian_ik_init_trajs(
    start:       jnp.ndarray,
    goal:        jnp.ndarray,
    n_batch:     int,
    n_timesteps: int,
    key:         jnp.ndarray,
    robot:       Robot,
    robot_coll:  RobotCollisionSpherized,
    world_geoms: tuple,
    noise_scale: float = 0.05,
) -> jnp.ndarray:
    """Initialize trajectories via Cartesian interpolation + batched MPPI IK.

    Steps:
      1. FK at start/goal to get end-effector SE(3) poses.
      2. Interpolate n_timesteps poses in Cartesian space (position lerp +
         SO(3) log/exp SLERP).
      3. Solve all T IK problems in one MPPI CUDA batch call, warm-started
         from linear joint-space interpolation, with self + world collision
         penalty as a constraint.
      4. Tile the resulting base trajectory to [B, T, DOF] and add noise.
    """
    ee_idx = robot.links.names.index(EE_LINK_NAME)

    # Step 1 — FK
    start_pose = jaxlie.SE3(robot.forward_kinematics(start)[ee_idx])
    goal_pose  = jaxlie.SE3(robot.forward_kinematics(goal)[ee_idx])

    # Step 2 — Cartesian interpolation
    start_rot = start_pose.rotation()
    goal_rot  = goal_pose.rotation()
    rel_log   = (start_rot.inverse() @ goal_rot).log()   # SO(3) tangent [3]
    start_t   = start_pose.translation()
    goal_t    = goal_pose.translation()

    alphas = jnp.linspace(0.0, 1.0, n_timesteps)

    def _interp(alpha):
        rot = start_rot @ jaxlie.SO3.exp(alpha * rel_log)
        pos = start_t + alpha * (goal_t - start_t)
        return jaxlie.SE3.from_rotation_and_translation(rot, pos)

    interp_poses = jax.vmap(_interp)(alphas)   # SE3 batch of shape (T,)

    # Warm-start: linear joint-space interpolation
    t_grid    = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]
    prev_cfgs = start[None] * (1.0 - t_grid) + goal[None] * t_grid   # [T, DOF]

    # Step 3 — Collision-free IK
    # Capture robot_coll and world_geoms as a closure; only the dummy arg is
    # traced by JAX so no retracing occurs when obstacle geometry is unchanged.
    def _collision_penalty(cfg, robot, _):
        cost = jnp.zeros(())
        self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
        cost += jnp.sum(-jnp.minimum(colldist_from_sdf(self_dists, 0.01), 0.0))
        for wg in world_geoms:
            wd = robot_coll.compute_world_collision_distance(robot, cfg, wg)
            cost += jnp.sum(-jnp.minimum(colldist_from_sdf(wd, 0.01), 0.0))
        return cost

    key, subkey = jax.random.split(key)
    base_traj = mppi_ik_solve_cuda_batch(
        robot               = robot,
        target_link_indices = ee_idx,
        target_poses        = interp_poses,
        rng_key             = subkey,
        previous_cfgs       = prev_cfgs,
        num_seeds           = 8,
        n_particles         = 64,
        n_mppi_iters        = 20,
        n_lbfgs_iters       = 25,
        m_lbfgs             = 5,
        constraints         = [_collision_penalty],
        constraint_args     = [jnp.zeros(())],   # dummy; fn closes over geoms
        constraint_weights  = [1e4],
    )   # [T, DOF]

    # Step 4 — Tile + noise → [B, T, DOF]
    trajs = jnp.broadcast_to(base_traj[None], (n_batch, n_timesteps, base_traj.shape[-1]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise


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
    return f"  {name:<26s}  {elapsed:>10.3f}  {smoothness:>12.4f}  {solved:>6d}/{BATCH_SIZE}"


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

    # Pass all geometry types to sco_trajopt so the optimizer sees all obstacles.

    # --- Build Cartesian-space IK initial trajectories ----------------------
    print(f"\nBuilding Cartesian IK initial trajectories (B={BATCH_SIZE}, T={N_TIMESTEPS})...")
    key = jax.random.PRNGKey(0)

    init_trajs = make_cartesian_ik_init_trajs(
        start       = start,
        goal        = goal,
        n_batch     = BATCH_SIZE,
        n_timesteps = N_TIMESTEPS,
        key         = key,
        robot       = robot,
        robot_coll  = robot_coll,
        world_geoms = world_geoms,
        noise_scale = NOISE_SCALE,
    )                                    # [B, T, DOF]
    print(f"  init_trajs shape : {init_trajs.shape}")

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

    # --- Warm-up JIT compile (JAX) -------------------------------------------
    print("\n[JAX] Warm-up / JIT compilation (first call)...")
    t0 = time.perf_counter()
    best_traj_warmup, costs_warmup, final_trajs_warmup = sco_trajopt(
        init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
    )
    best_traj_warmup.block_until_ready()
    jax_compile_time = time.perf_counter() - t0
    print(f"  Compile + first run : {jax_compile_time:.3f} s")

    # --- Timed benchmark run (JAX) -------------------------------------------
    print("\n[JAX] Timed benchmark run (second call)...")
    t0 = time.perf_counter()
    best_traj, costs, final_trajs = sco_trajopt(
        init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
    )
    best_traj.block_until_ready()
    jax_elapsed = time.perf_counter() - t0

    # --- JAX metrics ---------------------------------------------------------
    jax_smoothness = float(jnp.mean(
        jnp.array([compute_smoothness(final_trajs[b]) for b in range(BATCH_SIZE)])
    ))
    jax_n_solved = check_solved(final_trajs, goal, robot, robot_coll, list(world_geoms))
    jax_best_cost = float(jnp.min(costs))

    # --- Warm-up CUDA kernel (first call loads .so + runs kernel) ------------
    print("\n[CUDA] Warm-up / kernel load (first call)...")
    cuda_compile_time = None
    cuda_elapsed      = None
    cuda_smoothness   = None
    cuda_n_solved     = None
    cuda_best_cost    = None
    try:
        t0 = time.perf_counter()
        best_traj_cu_wu, _, _ = sco_trajopt(
            init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg,
            use_cuda=True,
        )
        best_traj_cu_wu.block_until_ready()
        cuda_compile_time = time.perf_counter() - t0
        print(f"  First run (kernel load) : {cuda_compile_time:.3f} s")

        # --- Timed benchmark run (CUDA) --------------------------------------
        print("\n[CUDA] Timed benchmark run (second call)...")
        t0 = time.perf_counter()
        best_traj_cu, costs_cu, final_trajs_cu = sco_trajopt(
            init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg,
            use_cuda=True,
        )
        best_traj_cu.block_until_ready()
        cuda_elapsed = time.perf_counter() - t0

        cuda_smoothness = float(jnp.mean(
            jnp.array([compute_smoothness(final_trajs_cu[b]) for b in range(BATCH_SIZE)])
        ))
        cuda_n_solved = check_solved(
            final_trajs_cu, goal, robot, robot_coll, list(world_geoms)
        )
        cuda_best_cost = float(jnp.min(costs_cu))

    except Exception as exc:
        print(f"  CUDA benchmark skipped: {exc}")

    # --- Results table -------------------------------------------------------
    print("\nComputing metrics...")
    header = (
        "\n"
        f"  {'Method':<26s}  {'Time (s)':>10s}  {'Smoothness':>12s}  {'Solved':>8s}\n"
        f"  {'-'*26}  {'-'*10}  {'-'*12}  {'-'*8}"
    )
    print(header)
    print(_fmt_row("JAX SCO-TrajOpt (JIT)", jax_elapsed, jax_smoothness, jax_n_solved))
    if cuda_elapsed is not None:
        print(_fmt_row("CUDA SCO-TrajOpt", cuda_elapsed, cuda_smoothness, cuda_n_solved))

    print(f"\n  JAX  best trajectory cost      : {jax_best_cost:.4f}")
    if cuda_best_cost is not None:
        print(f"  CUDA best trajectory cost      : {cuda_best_cost:.4f}")
        speedup = jax_elapsed / cuda_elapsed if cuda_elapsed > 0 else float("nan")
        print(f"  CUDA speedup (timed runs)      : {speedup:.2f}x")
    print(f"\n  JAX  compile + first-run time  : {jax_compile_time:.3f} s")
    if cuda_compile_time is not None:
        print(f"  CUDA kernel load + first run   : {cuda_compile_time:.3f} s")
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
