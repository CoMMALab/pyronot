"""Benchmark: compare SCO, LS, CHOMP, and STOMP TrajOpt on a VAMP problem instance.

Loads a planning problem from the VAMP dataset (panda/problems.pkl), runs FK
for Cartesian start/goal poses, seeds trajectories through the existing
Cartesian spline + batched IK pipeline, then benchmarks all solvers on the
same seeded batches.

Usage
-----
    python tests/bench_trajopt.py [--problem bookshelf_tall] [--index 1]
    python tests/bench_trajopt.py --disable chomp --disable stomp

The script prints a results table with columns:
    Method | Seed (s) | Warmup (s) | Run (s) | TrajOpt (s) | Smoothness | Solved (B=25)

"Solved" counts trajectories whose final waypoint reaches the Cartesian goal
and stays collision-free for all timesteps.
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
from pyronot.optimization_engines import (
    ChompTrajOptConfig,
    LsTrajOptConfig,
    ScoTrajOptConfig,
    StompTrajOptConfig,
    chomp_trajopt,
    ls_trajopt,
    sco_trajopt,
    stomp_trajopt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 25
N_TIMESTEPS = 64
NOISE_SCALE = 0.05   # std of per-seed perturbation
EE_LINK_NAME = "panda_hand"
GOAL_POS_TOL = 0.01  # m — EE position must be within this of goal pose
GOAL_ORI_TOL = 0.05  # rad — EE orientation must be within this of goal pose
COLL_TOL = 0.0       # signed-distance threshold to count as collision-free

RESOURCES = Path(__file__).parent.parent / "resources"
PANDA_URDF = RESOURCES / "panda" / "panda_spherized.urdf"
PANDA_SRDF = RESOURCES / "panda" / "panda.srdf"
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
    robot = Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=PANDA_SRDF.as_posix())
    return robot, robot_coll, urdf


def compute_smoothness(traj: jnp.ndarray) -> float:
    """Mean squared velocity norm over the trajectory."""
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
    """Count trajectories that reach the goal pose without collisions."""
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


def _fmt_row(name: str, seed_time: float, warmup: float, elapsed: float, trajopt_time: float, smoothness: float, solved: int) -> str:
    return (
        f"  {name:<28s}  {seed_time:>10.3f}  {warmup:>10.3f}  {elapsed:>10.3f}"
        f"  {trajopt_time:>12.3f}  {smoothness:>12.4f}  {solved:>6d}/{BATCH_SIZE}"
    )


def _seed_trajectories_once(
    motion_gen: TrajoptMotionGenerator,
    control_poses: jaxlie.SE3,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Seed trajectories once and return seed batch + endpoint configs + elapsed."""
    t0 = time.perf_counter()
    init_trajs, start_cfg, goal_cfg = motion_gen._seed_trajectories(control_poses, key)
    init_trajs.block_until_ready()
    elapsed = time.perf_counter() - t0
    return init_trajs, start_cfg, goal_cfg, elapsed


def _run_solver(
    name: str,
    solver_fn,
    solver_cfg,
    init_trajs: jnp.ndarray,
    start_cfg: jnp.ndarray,
    goal_cfg: jnp.ndarray,
    robot: Robot,
    robot_coll: RobotCollisionSpherized,
    world_geoms: tuple,
    goal_pose: jaxlie.SE3,
    ee_idx: int,
    seed_time: float,
    *,
    use_cuda: bool,
    solver_kwargs: dict | None = None,
) -> dict:
    """Run warm-up + timed solver call and compute quality metrics."""
    extra = solver_kwargs or {}

    t0_wu = time.perf_counter()
    best_wu, _, _ = solver_fn(
        init_trajs,
        start_cfg,
        goal_cfg,
        robot,
        robot_coll,
        world_geoms,
        solver_cfg,
        use_cuda=use_cuda,
        **extra,
    )
    best_wu.block_until_ready()
    warmup_elapsed = time.perf_counter() - t0_wu

    t0 = time.perf_counter()
    best_traj, costs, final_trajs = solver_fn(
        init_trajs,
        start_cfg,
        goal_cfg,
        robot,
        robot_coll,
        world_geoms,
        solver_cfg,
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
        "elapsed": trajopt_time,
        "trajopt_time": trajopt_time,
        "smoothness": smoothness,
        "n_solved": n_solved,
        "best_cost": float(jnp.min(costs)),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main(problem_name: str, index: int, disabled_solvers: set[str]) -> None:
    enabled = [s for s in ("sco", "ls", "chomp", "stomp") if s not in disabled_solvers]
    print(
        "\n=== TrajOpt Benchmark"
        f"  |  problem={problem_name!r}  index={index}"
        f"  |  enabled={enabled} ===\n"
    )

    print("Loading problem...")
    problem_data = load_problem(problem_name, index)
    start_cfg = jnp.array(problem_data["start"], dtype=jnp.float32)
    goals = jnp.array(problem_data["goals"], dtype=jnp.float32)
    goal_cfg = goals[0]
    valid = problem_data.get("valid", True)
    print(f"  Start cfg : {np.array(start_cfg)}")
    print(f"  Goal  cfg : {np.array(goal_cfg)}")
    print(f"  Valid     : {valid}")

    print("\nLoading robot...")
    robot, robot_coll, _ = load_robot(PANDA_URDF)
    dof = robot.joints.num_actuated_joints
    ee_idx = robot.links.names.index(EE_LINK_NAME)
    print(f"  DOF   : {dof}")

    print("\nRunning FK for start/goal Cartesian poses...")
    start_pose = jaxlie.SE3(robot.forward_kinematics(start_cfg)[ee_idx])
    goal_pose = jaxlie.SE3(robot.forward_kinematics(goal_cfg)[ee_idx])
    print(f"  Start pos : {np.array(start_pose.translation())}")
    print(f"  Goal  pos : {np.array(goal_pose.translation())}")

    mid_pos = (start_pose.translation() + goal_pose.translation()) / 2.0
    mid_rot = start_pose.rotation() @ jaxlie.SO3.exp(
        0.5 * (start_pose.rotation().inverse() @ goal_pose.rotation()).log()
    )
    mid_pose = jaxlie.SE3.from_rotation_and_translation(mid_rot, mid_pos)

    print("\nBuilding collision environment...")
    obstacles = create_collision_environment(problem_data)
    world_geoms = stack_obstacles(obstacles)
    print(f"  Geometry groups : {len(world_geoms)}")
    for wg in world_geoms:
        print(f"    {type(wg).__name__}  batch={wg.get_batch_axes()}")

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
    chomp_cfg = ChompTrajOptConfig(
        n_iters=100,
        step_size=0.2,
        w_smooth=1.0,
        w_vel=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        collision_margin=0.02,
        w_limits=1.0,
        use_covariant_update=True,
        smoothness_reg=1e-3,
    )
    ls_cfg = LsTrajOptConfig(
        n_outer_iters=10,
        n_ls_iters=10,
        lambda_init=5e-3,
        w_smooth=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        w_collision_max=100.0,
        penalty_scale=3.0,
        collision_margin=0.02,
        w_trust=0.5,
        w_limits=1.0,
        w_endpoint=100.0,
        smooth_min_temperature=0.05,
        max_delta_per_step=0.1,
    )
    stomp_cfg = StompTrajOptConfig(
        # Throughput-oriented setting: more parallel samples, fewer iterations.
        n_iters=40,
        n_samples=96,
        noise_scale=0.03,
        # temperature=1.0 is meaningful with use_cost_normalization=True:
        # a 1-std cost difference between samples → exp(-1) ≈ 0.37 weight ratio.
        temperature=1.0,
        # step_size acts as an EMA blend factor (cuRobo step_size_mean).
        # 0.3 prevents trajectory oscillation while still allowing convergence.
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
        # Both enabled by default; listed explicitly for documentation.
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

    key = jax.random.PRNGKey(0)
    results = []

    for mode in ("linear", "cubic", "bspline"):
        if mode == "linear":
            wp = None
            label_suffix = "(K=2)"
            control_wxyz_xyz = jnp.concatenate(
                [start_pose.wxyz_xyz[None], goal_pose.wxyz_xyz[None]], axis=0
            )
        else:
            wp = mid_pose
            label_suffix = "(K=3)"
            control_wxyz_xyz = jnp.concatenate(
                [start_pose.wxyz_xyz[None], mid_pose.wxyz_xyz[None], goal_pose.wxyz_xyz[None]],
                axis=0,
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
            cartesian_spline_mode=mode,
            trajopt_cfg=sco_cfg,
            use_cuda=False,
        )

        key, seed_key_wu = jax.random.split(key)
        _seed_trajectories_once(motion_gen, control_poses, seed_key_wu)

        key, seed_key = jax.random.split(key)
        init_trajs, seeded_start_cfg, seeded_goal_cfg, seed_time = _seed_trajectories_once(
            motion_gen, control_poses, seed_key
        )

        runs = []
        if "sco" not in disabled_solvers:
            runs.append(("SCO", sco_trajopt, sco_cfg, {}))
        if "ls" not in disabled_solvers:
            runs.append(("LS", ls_trajopt, ls_cfg, {"key": jax.random.PRNGKey(123)}))
        if "chomp" not in disabled_solvers:
            runs.append(("CHOMP", chomp_trajopt, chomp_cfg, {}))
        if "stomp" not in disabled_solvers:
            runs.append(("STOMP", stomp_trajopt, stomp_cfg, {"key": jax.random.PRNGKey(42)}))

        if not runs:
            raise ValueError("All solvers are disabled. Enable at least one solver.")

        backends = (
            ("JAX", False),
            ("CUDA", True),
        )

        for solver_name, solver_fn, solver_cfg, solver_kwargs in runs:
            for backend_name, use_cuda in backends:
                print(
                    f"\n[{mode} {label_suffix}] {solver_name} {backend_name} "
                    "— warm-up + timed run..."
                )
                try:
                    res = _run_solver(
                        f"{solver_name:<5s} {backend_name:<4s} {mode:>8s} {label_suffix}",
                        solver_fn,
                        solver_cfg,
                        init_trajs,
                        seeded_start_cfg,
                        seeded_goal_cfg,
                        robot,
                        robot_coll,
                        world_geoms,
                        goal_pose,
                        ee_idx,
                        seed_time,
                        use_cuda=use_cuda,
                        solver_kwargs=solver_kwargs,
                    )
                    results.append(res)
                except Exception as exc:
                    print(f"  Skipping {solver_name} {backend_name}: {exc}")

    print("\n" + "=" * 92)
    header = (
        f"  {'Method':<28s}  {'Seed (s)':>10s}  {'Warmup (s)':>10s}  {'Run (s)':>10s}"
        f"  {'TrajOpt (s)':>12s}  {'Smoothness':>12s}  {'Solved':>8s}\n"
        f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}"
    )
    print(header)
    for r in results:
        print(_fmt_row(
            r["name"], r["seed_time"], r["warmup"], r["elapsed"],
            r["trajopt_time"], r["smoothness"], r["n_solved"],
        ))

    print()
    for r in results:
        print(f"  {r['name']:<28s}  best_cost = {r['best_cost']:.4f}")

    print(f"\n  Batch size (B)                 : {BATCH_SIZE}")
    print(f"  Timesteps (T)                  : {N_TIMESTEPS}")
    print(f"  DOF                            : {dof}")
    print(f"  SCO outer iterations           : {sco_cfg.n_outer_iters}")
    print(f"  SCO inner L-BFGS iters         : {sco_cfg.n_inner_iters}")
    if "ls" not in disabled_solvers:
        print(f"  LS outer iterations            : {ls_cfg.n_outer_iters}")
        print(f"  LS inner LM iters              : {ls_cfg.n_ls_iters}")
    if "chomp" not in disabled_solvers:
        print(f"  CHOMP iterations               : {chomp_cfg.n_iters}")
    if "stomp" not in disabled_solvers:
        print(f"  STOMP iterations               : {stomp_cfg.n_iters}")
        print(f"  STOMP samples per iter         : {stomp_cfg.n_samples}")
        print(f"  STOMP noise scale              : {stomp_cfg.noise_scale}")
        print(f"  STOMP temperature              : {stomp_cfg.temperature}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SCO/LS/CHOMP/STOMP TrajOpt on a VAMP problem.")
    parser.add_argument("--problem", default="bookshelf_tall", help="Problem name in problems.pkl")
    parser.add_argument("--index", type=int, default=1, help="Problem instance index")
    parser.add_argument(
        "--disable",
        action="append",
        choices=("sco", "ls", "chomp", "stomp"),
        default=[],
        help=(
            "Disable one or more solvers. Repeatable. "
            "Default disables: none."
        ),
    )
    args = parser.parse_args()
    main(args.problem, args.index, set(args.disable))
