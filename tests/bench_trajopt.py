"""Benchmark: compare SCO, LS, CHOMP, STOMP, and cuRobo TrajOpt on a VAMP problem instance.

Loads a planning problem from the VAMP dataset (panda/problems.pkl), runs FK
for Cartesian start/goal poses, seeds trajectories through the existing
Cartesian spline + batched IK pipeline, then benchmarks all solvers on the
same seeded batches.

Usage
-----
    python tests/bench_trajopt.py [--problem bookshelf_tall] [--index 1]
    python tests/bench_trajopt.py --disable chomp --disable stomp
    python tests/bench_trajopt.py --disable sco --disable ls --disable chomp --disable stomp
    # (runs cuRobo only)

The script prints a results table with columns:
    Method | Seed (s) | Warmup (s) | Run (s) | TrajOpt (s) | Smoothness | Solved (B=25)

"Solved" counts trajectories whose final waypoint reaches the Cartesian goal
and stays collision-free for all timesteps.
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


# ---------------------------------------------------------------------------
# cuRobo baseline
# ---------------------------------------------------------------------------

def _build_curobo_world(problem_data: dict) -> dict:
    """Convert VAMP obstacle dicts to a cuRobo world_model config dict.

    cuRobo expects each geometry type as a dict keyed by obstacle name, e.g.:
        {"cuboid": {"box_0": {"pose": [...], "dims": [...]}, ...}}
    Pose format: [x, y, z, qw, qx, qy, qz].
    """
    cuboid_dict: dict = {}
    sphere_dict: dict = {}
    cylinder_dict: dict = {}

    for i, s in enumerate(problem_data.get("sphere", [])):
        name = s.get("name", f"sphere_{i}")
        sphere_dict[name] = {
            "pose": s["position"] + [1.0, 0.0, 0.0, 0.0],
            "radius": float(s["radius"]),
        }

    for i, b in enumerate(problem_data.get("box", [])):
        name = b.get("name", f"box_{i}")
        he = b["half_extents"]
        dims = [2.0 * he[0], 2.0 * he[1], 2.0 * he[2]]
        if "orientation_quat_xyzw" in b and b["orientation_quat_xyzw"] is not None:
            q = b["orientation_quat_xyzw"]  # [x,y,z,w]
            pose = b["position"] + [q[3], q[0], q[1], q[2]]
        else:
            pose = b["position"] + [1.0, 0.0, 0.0, 0.0]
        cuboid_dict[name] = {"pose": pose, "dims": dims}

    for i, c in enumerate(problem_data.get("cylinder", [])):
        name = c.get("name", f"cyl_{i}")
        if "orientation_quat_xyzw" in c and c["orientation_quat_xyzw"] is not None:
            q = c["orientation_quat_xyzw"]
            pose = c["position"] + [q[3], q[0], q[1], q[2]]
        else:
            pose = c["position"] + [1.0, 0.0, 0.0, 0.0]
        cylinder_dict[name] = {
            "pose": pose,
            "radius": float(c["radius"]),
            "height": float(c["length"]),
        }

    world_cfg: dict = {}
    if cuboid_dict:
        world_cfg["cuboid"] = cuboid_dict
    if sphere_dict:
        world_cfg["sphere"] = sphere_dict
    if cylinder_dict:
        world_cfg["cylinder"] = cylinder_dict
    return world_cfg


def run_curobo_baseline(
    problem_data: dict,
    start_cfg_np: np.ndarray,
    goal_pose: "jaxlie.SE3",
    ee_idx: int,
    robot: "Robot",
    robot_coll: "RobotCollisionSpherized",
    world_geoms: list,
) -> dict:
    """Run cuRobo MotionGen on the same problem and report comparable metrics."""
    # Release JAX GPU memory before allocating cuRobo buffers
    jax.clear_caches()

    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
    world_cfg = _build_curobo_world(problem_data)

    # cuRobo's Warp kernels can exceed CUDA register limits at high timestep counts
    # with many obstacles.  Cap at 48 (cuRobo interpolates the result anyway).
    curobo_tsteps = min(N_TIMESTEPS, 48)
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        world_model=world_cfg if world_cfg else None,
        tensor_args=tensor_args,
        num_trajopt_seeds=4,
        num_ik_seeds=32,
        trajopt_tsteps=curobo_tsteps,
    )
    motion_gen = MotionGen(motion_gen_cfg)
    B = BATCH_SIZE

    # Warm-up (JIT compile) — disable graph to avoid goal-type mismatch with plan_batch
    t0_wu = time.perf_counter()
    motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False, batch=B)
    torch.cuda.synchronize()
    warmup_elapsed = time.perf_counter() - t0_wu

    # Build batched goal pose and start state — same goal for all B problems
    gp_trans = np.array(goal_pose.translation())
    gp_wxyz = np.array(goal_pose.rotation().wxyz)
    goal_position = torch.tensor(gp_trans, device="cuda:0", dtype=torch.float32).unsqueeze(0).expand(B, -1)
    goal_quaternion = torch.tensor(gp_wxyz, device="cuda:0", dtype=torch.float32).unsqueeze(0).expand(B, -1)
    cu_goal = CuPose(position=goal_position.contiguous(), quaternion=goal_quaternion.contiguous())

    start_tensor = torch.tensor(start_cfg_np, device="cuda:0", dtype=torch.float32).unsqueeze(0).expand(B, -1)
    start_js = JointState.from_position(start_tensor.contiguous())

    plan_cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=1)

    # Timed run (batch planning)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = motion_gen.plan_batch(start_js, cu_goal, plan_cfg)
    torch.cuda.synchronize()
    trajopt_time = time.perf_counter() - t0

    # Extract trajectories for evaluation with pyronot's checker
    success_mask = result.success.cpu()  # (B,)
    n_success = int(success_mask.sum().item())
    if n_success > 0:
        # optimized_plan.position: (B, T, dof)
        trajs_torch = result.optimized_plan.position
        trajs_np = trajs_torch.cpu().numpy()
        trajs_jnp = jnp.array(trajs_np)
        smoothness = float(jnp.mean(jnp.array([
            compute_smoothness(trajs_jnp[b]) for b in range(B) if success_mask[b]
        ])))
        n_solved = check_solved(
            trajs_jnp,
            goal_pose, ee_idx, robot, robot_coll, world_geoms,
        )
    else:
        smoothness = float("inf")
        n_solved = 0

    return {
        "name": "cuRobo",
        "seed_time": 0.0,
        "warmup": warmup_elapsed,
        "elapsed": trajopt_time,
        "trajopt_time": trajopt_time,
        "smoothness": smoothness,
        "n_solved": n_solved,
        "best_cost": float(result.optimized_dt[success_mask].min().item()) if n_success > 0 else float("inf"),
    }


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
    enabled = [s for s in ("sco", "ls", "lbfgs", "chomp", "stomp", "curobo") if s not in disabled_solvers]
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
    lbfgs_cfg = LbfgsTrajOptConfig(
        n_iters=100,
        m_lbfgs=6,
        w_smooth=1.0,
        w_vel=1.0,
        w_acc=0.5,
        w_jerk=0.1,
        w_collision=10.0,
        w_collision_max=100.0,
        penalty_scale=2.0,
        escalation_interval=20,
        collision_margin=0.02,
        w_limits=1.0,
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
        if "lbfgs" not in disabled_solvers:
            runs.append(("L-BFGS", lbfgs_trajopt, lbfgs_cfg, {}))
        if "stomp" not in disabled_solvers:
            runs.append(("STOMP", stomp_trajopt, stomp_cfg, {"key": jax.random.PRNGKey(42)}))

        if not runs:
            continue  # skip spline modes when only cuRobo is enabled

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

    # ----- cuRobo baseline -----
    if "curobo" not in disabled_solvers:
        print("\n[cuRobo] warm-up + timed run...")
        try:
            curobo_res = run_curobo_baseline(
                problem_data,
                np.array(start_cfg),
                goal_pose,
                ee_idx,
                robot,
                robot_coll,
                list(world_geoms),
            )
            results.append(curobo_res)
        except Exception as exc:
            print(f"  Skipping cuRobo: {exc}")

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
    if "lbfgs" not in disabled_solvers:
        print(f"  L-BFGS iterations              : {lbfgs_cfg.n_iters}")
        print(f"  L-BFGS escalation interval     : {lbfgs_cfg.escalation_interval}")
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
    parser.add_argument("--problem", default="bookshelf_thin", help="Problem name in problems.pkl")
    parser.add_argument("--index", type=int, default=1, help="Problem instance index")
    parser.add_argument(
        "--disable",
        action="append",
        choices=("sco", "ls", "lbfgs", "chomp", "stomp", "curobo"),
        default=[],
        help=(
            "Disable one or more solvers. Repeatable. "
            "Default disables: none."
        ),
    )
    args = parser.parse_args()
    main(args.problem, args.index, set(args.disable))
