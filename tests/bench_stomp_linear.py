"""Short benchmark: STOMP-only trajectory optimization with linear spline seeding.

This is a reduced version of ``tests/bench_trajopt.py`` that only runs:
  - spline mode: linear (K=2 control poses)
  - solver: STOMP
  - backends: JAX and CUDA

Usage
-----
    python tests/bench_stomp_linear.py [--problem bookshelf_tall] [--index 1]
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyronot._robot import Robot
from pyronot.collision._obstacles import create_collision_environment, stack_obstacles
from pyronot.collision._robot_collision import RobotCollisionSpherized
from pyronot.motion_generators import TrajoptMotionGenerator
from pyronot.optimization_engines import (
    ScoTrajOptConfig,
    StompTrajOptConfig,
    stomp_trajopt,
)


BATCH_SIZE = 25
N_TIMESTEPS = 64
NOISE_SCALE = 0.05
EE_LINK_NAME = "panda_hand"
GOAL_POS_TOL = 0.01
GOAL_ORI_TOL = 0.05
COLL_TOL = 0.0

RESOURCES = Path(__file__).parent.parent / "resources"
PANDA_URDF = RESOURCES / "panda" / "panda_spherized.urdf"
PANDA_SRDF = RESOURCES / "panda" / "panda.srdf"
PROBLEMS_PKL = RESOURCES / "panda" / "problems.pkl"


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
    robot = Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=PANDA_SRDF.as_posix())
    return robot, robot_coll, urdf


def compute_smoothness(traj: jnp.ndarray) -> float:
    vel = traj[1:] - traj[:-1]
    return float(jnp.mean(jnp.sum(vel ** 2, axis=-1)))


def check_solved(
    trajs: jnp.ndarray,
    goal_pose: jaxlie.SE3,
    ee_idx: int,
    robot: Robot,
    robot_coll: RobotCollisionSpherized,
    world_geoms: list,
) -> int:
    n_solved = 0
    for b in range(trajs.shape[0]):
        traj = trajs[b]

        final_pose = jaxlie.SE3(robot.forward_kinematics(traj[-1])[ee_idx])
        pos_err = float(jnp.linalg.norm(final_pose.translation() - goal_pose.translation()))
        ori_err = float(jnp.linalg.norm((final_pose.rotation().inverse() @ goal_pose.rotation()).log()))
        if pos_err > GOAL_POS_TOL or ori_err > GOAL_ORI_TOL:
            continue

        collision_free = True
        for t in range(traj.shape[0]):
            cfg = traj[t]
            self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
            if float(jnp.min(self_dists)) < COLL_TOL:
                collision_free = False
                break

            for world_geom in world_geoms:
                world_dists = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
                if float(jnp.min(world_dists)) < COLL_TOL:
                    collision_free = False
                    break
            if not collision_free:
                break

        if collision_free:
            n_solved += 1

    return n_solved


def _fmt_row(name: str, seed_time: float, warmup: float, trajopt_time: float, smoothness: float, solved: int) -> str:
    return (
        f"  {name:<24s}  {seed_time:>10.3f}  {warmup:>10.3f}  {trajopt_time:>12.3f}"
        f"  {smoothness:>12.4f}  {solved:>6d}/{BATCH_SIZE}"
    )


def _seed_trajectories_once(
    motion_gen: TrajoptMotionGenerator,
    control_poses: jaxlie.SE3,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    t0 = time.perf_counter()
    init_trajs, start_cfg, goal_cfg = motion_gen._seed_trajectories(control_poses, key)
    init_trajs.block_until_ready()
    elapsed = time.perf_counter() - t0
    return init_trajs, start_cfg, goal_cfg, elapsed


def run_stomp(
    name: str,
    init_trajs: jnp.ndarray,
    start_cfg: jnp.ndarray,
    goal_cfg: jnp.ndarray,
    robot: Robot,
    robot_coll: RobotCollisionSpherized,
    world_geoms: tuple,
    goal_pose: jaxlie.SE3,
    ee_idx: int,
    seed_time: float,
    stomp_cfg: StompTrajOptConfig,
    use_cuda: bool,
) -> dict:
    extra = {"key": jax.random.PRNGKey(42)}

    t0_wu = time.perf_counter()
    best_wu, _, _ = stomp_trajopt(
        init_trajs,
        start_cfg,
        goal_cfg,
        robot,
        robot_coll,
        world_geoms,
        stomp_cfg,
        use_cuda=use_cuda,
        **extra,
    )
    best_wu.block_until_ready()
    warmup_elapsed = time.perf_counter() - t0_wu

    t0 = time.perf_counter()
    best_traj, costs, final_trajs = stomp_trajopt(
        init_trajs,
        start_cfg,
        goal_cfg,
        robot,
        robot_coll,
        world_geoms,
        stomp_cfg,
        use_cuda=use_cuda,
        **extra,
    )
    best_traj.block_until_ready()
    trajopt_time = time.perf_counter() - t0

    smoothness = float(jnp.mean(jnp.array([
        compute_smoothness(final_trajs[b]) for b in range(BATCH_SIZE)
    ])))
    n_solved = check_solved(
        final_trajs,
        goal_pose,
        ee_idx,
        robot,
        robot_coll,
        list(world_geoms),
    )

    return {
        "name": name,
        "seed_time": seed_time,
        "warmup": warmup_elapsed,
        "trajopt_time": trajopt_time,
        "smoothness": smoothness,
        "n_solved": n_solved,
        "best_cost": float(jnp.min(costs)),
    }


def main(problem_name: str, index: int) -> None:
    print(f"\n=== STOMP Linear TrajOpt Benchmark  |  problem={problem_name!r}  index={index} ===\n")

    problem_data = load_problem(problem_name, index)
    start_cfg = jnp.array(problem_data["start"], dtype=jnp.float32)
    goals = jnp.array(problem_data["goals"], dtype=jnp.float32)
    goal_cfg = goals[0]

    robot, robot_coll, _ = load_robot(PANDA_URDF)
    ee_idx = robot.links.names.index(EE_LINK_NAME)

    start_pose = jaxlie.SE3(robot.forward_kinematics(start_cfg)[ee_idx])
    goal_pose = jaxlie.SE3(robot.forward_kinematics(goal_cfg)[ee_idx])

    obstacles = create_collision_environment(problem_data)
    world_geoms = stack_obstacles(obstacles)

    sco_cfg = ScoTrajOptConfig(
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
    stomp_cfg = StompTrajOptConfig(
        n_iters=40,
        n_samples=96,
        noise_scale=0.03,
        temperature=1.0,
        step_size=0.3,
        w_smooth=1.0,
        w_vel=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        w_collision_max=100.0,
        collision_penalty_scale=1.05,
        collision_margin=0.02,
        w_limits=1.0,
        use_covariant_update=False,
        smoothness_reg=0.1,
        use_cost_normalization=True,
        use_null_particle=True,
        use_elite_filter=True,
        elite_frac=0.25,
        adaptive_covariance=True,
        cov_update_rate=0.2,
        noise_decay=0.99,
        noise_scale_min=0.003,
        noise_scale_max=0.1,
        normalize_smooth_noise_scale=True,
        n_lbfgs_iters=10,
        m_lbfgs=5,
        lbfgs_step_scale=1.0,
    )

    control_wxyz_xyz = jnp.concatenate(
        [start_pose.wxyz_xyz[None], goal_pose.wxyz_xyz[None]], axis=0
    )
    control_poses = jaxlie.SE3(control_wxyz_xyz)

    motion_gen = TrajoptMotionGenerator(
        robot=robot,
        robot_coll=robot_coll,
        world_geoms=world_geoms,
        ee_link_name=EE_LINK_NAME,
        n_timesteps=N_TIMESTEPS,
        n_batch=BATCH_SIZE,
        noise_scale=NOISE_SCALE,
        cartesian_spline_mode="linear",
        trajopt_cfg=sco_cfg,
        use_cuda=False,
    )

    key = jax.random.PRNGKey(0)
    key, seed_key_wu = jax.random.split(key)
    _seed_trajectories_once(motion_gen, control_poses, seed_key_wu)

    key, seed_key = jax.random.split(key)
    init_trajs, seeded_start_cfg, seeded_goal_cfg, seed_time = _seed_trajectories_once(
        motion_gen, control_poses, seed_key
    )

    results = []
    for backend_name, use_cuda in (("JAX", False), ("CUDA", True)):
        print(f"Running STOMP {backend_name} (linear, K=2) ...")
        try:
            r = run_stomp(
                f"STOMP {backend_name:<4s} linear (K=2)",
                init_trajs,
                seeded_start_cfg,
                seeded_goal_cfg,
                robot,
                robot_coll,
                world_geoms,
                goal_pose,
                ee_idx,
                seed_time,
                stomp_cfg,
                use_cuda,
            )
            results.append(r)
        except Exception as exc:
            print(f"  Skipping STOMP {backend_name}: {exc}")

    print("\n" + "=" * 88)
    print(
        f"  {'Method':<24s}  {'Seed (s)':>10s}  {'Warmup (s)':>10s}  {'TrajOpt (s)':>12s}"
        f"  {'Smoothness':>12s}  {'Solved':>8s}\n"
        f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}"
    )
    for r in results:
        print(_fmt_row(
            r["name"],
            r["seed_time"],
            r["warmup"],
            r["trajopt_time"],
            r["smoothness"],
            r["n_solved"],
        ))

    print()
    for r in results:
        print(f"  {r['name']:<24s}  best_cost = {r['best_cost']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Short STOMP-only linear-spline benchmark.")
    parser.add_argument("--problem", default="bookshelf_tall", help="Problem name in problems.pkl")
    parser.add_argument("--index", type=int, default=1, help="Problem instance index")
    args = parser.parse_args()
    main(args.problem, args.index)
