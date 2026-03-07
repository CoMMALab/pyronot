"""Benchmark and correctness evaluation: JAX vs CUDA HJCD-IK solvers.

Measures for each solver across a set of randomised target poses:
  - Median solve time (ms)
  - Position error (mm)  — ‖t_actual − t_target‖₂
  - Rotation error (rad) — ‖log(R_target⁻¹ R_actual)‖₂
  - Agreement between JAX and CUDA solutions (joint-space Δq and task-space)

Usage:
    python tests/bench_ik.py

Prerequisites:
    1. A CUDA-capable GPU.
    2. CUDA IK library compiled:
           bash src/pyronot/cuda_kernels/build_ik_cuda.sh
    3. robot_descriptions installed:
           pip install robot_descriptions
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from pyronot.optimization_engines._hjcd_ik import hjcd_solve, hjcd_solve_cuda

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAME       = "panda"
TARGET_LINK_NAME = "panda_hand"

# Joints to keep fixed during IK (finger joints for the Panda).
FIXED_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

N_TARGETS  = 2000    # number of random target poses to evaluate
N_WARMUP   = 3     # JIT / kernel warm-up calls (discarded from timing)
N_TIMED    = 20    # timed repetitions per pose (median reported)

# IK solver hyper-parameters.
# Keep defaults on the fast path for both solvers. CUDA currently supports
# only jacobian_cd for coarse search.
IK_KWARGS_COMMON = dict(
    num_seeds      = 32,
    coarse_max_iter= 20,
    lm_max_iter    = 40,
    lambda_init    = 1e-3,
    continuity_weight   = 0.0,
    limit_prior_weight  = 1e-4,
    kick_scale     = 0.02,
)
IK_KWARGS_JAX = dict(**IK_KWARGS_COMMON)
IK_KWARGS_CUDA = dict(**IK_KWARGS_COMMON)

# Agreement threshold between JAX and CUDA outputs.
AGREE_POS_MM  = 5.0   # mm  — task-space position agreement
AGREE_ORI_RAD = 0.5   # rad — task-space orientation agreement

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    cfg:      np.ndarray   # (n_act,)
    pos_err:  float        # metres
    rot_err:  float        # radians
    time_ms:  float        # median solve time (milliseconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_errors(
    robot: pk.Robot,
    cfg: jax.Array,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> tuple[float, float]:
    """Return (pos_error_m, rot_error_rad) for a solved configuration."""
    Ts = robot.forward_kinematics(cfg)
    actual = jaxlie.SE3(Ts[target_link_index])
    pos_err = float(jnp.linalg.norm(actual.translation() - target_pose.translation()))
    rot_err = float(jnp.linalg.norm(
        (target_pose.rotation().inverse() @ actual.rotation()).log()
    ))
    return pos_err, rot_err


def _time_solve(fn, *args, n: int = N_TIMED, **kwargs) -> tuple[jax.Array, float]:
    """Run *fn* n times, return (last_output, median_wall_time_seconds)."""
    times = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


# ---------------------------------------------------------------------------
# Per-pose solve (returns SolveResult)
# ---------------------------------------------------------------------------

def _solve_jax(
    robot: pk.Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
    previous_cfg: jax.Array,
    fixed_joint_mask: jax.Array,
    rng_key: jax.Array,
    jit_fn,
    n_timed: int,
) -> SolveResult:
    cfg, t = _time_solve(
        jit_fn,
        robot, target_link_index, target_pose, rng_key, previous_cfg,
        fixed_joint_mask=fixed_joint_mask,
        **IK_KWARGS_JAX,
        n=n_timed,
    )
    pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)
    return SolveResult(np.array(cfg), pos_err, rot_err, t * 1e3)


def _solve_cuda(
    robot: pk.Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
    previous_cfg: jax.Array,
    fixed_joint_mask: jax.Array,
    rng_key: jax.Array,
    n_timed: int,
) -> SolveResult:
    cfg, t = _time_solve(
        hjcd_solve_cuda,
        robot, target_link_index, target_pose, rng_key, previous_cfg,
        fixed_joint_mask=fixed_joint_mask,
        **IK_KWARGS_CUDA,
        n=n_timed,
    )
    pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)
    return SolveResult(np.array(cfg), pos_err, rot_err, t * 1e3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 76)
    print(f"HJCD-IK benchmark: JAX vs CUDA  (robot={ROBOT_NAME}, "
          f"n_targets={N_TARGETS}, n_timed={N_TIMED})")
    print("=" * 76)

    # ------------------------------------------------------------------
    # Load robot
    # ------------------------------------------------------------------
    print("\nLoading robot ...")
    urdf   = load_robot_description(f"{ROBOT_NAME}_description")
    robot  = pk.Robot.from_urdf(urdf)
    n_act  = robot.joints.num_actuated_joints
    target_link_index = robot.links.names.index(TARGET_LINK_NAME)

    fixed_joint_mask = jnp.array(
        [name in FIXED_JOINT_NAMES for name in robot.joints.actuated_names],
        dtype=jnp.int32,
    )

    print(f"  {n_act} actuated joints, target link: '{TARGET_LINK_NAME}'")
    print(f"  Fixed joints: {[n for n in FIXED_JOINT_NAMES if n in robot.joints.actuated_names]}")

    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)
    rng_np = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Generate target poses via FK on random configurations
    # ------------------------------------------------------------------
    print("\nGenerating target poses ...")
    target_cfgs_np = rng_np.uniform(lo, hi, size=(N_TARGETS, n_act)).astype(np.float32)
    target_poses: list[jaxlie.SE3] = []
    for i in range(N_TARGETS):
        cfg_i = jnp.array(target_cfgs_np[i])
        Ts    = robot.forward_kinematics(cfg_i)
        target_poses.append(jaxlie.SE3(Ts[target_link_index]))

    # ------------------------------------------------------------------
    # JIT-compile the JAX solver once (static args set by first call)
    # ------------------------------------------------------------------
    print("JIT-compiling JAX solver ...")
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)
    rng0    = jax.random.PRNGKey(0)

    # Wrap with fixed static kwargs so jit sees the same static signature.
    import functools
    jit_solve = jax.jit(
        functools.partial(hjcd_solve, **IK_KWARGS_JAX),
        static_argnames=("target_link_index", "num_seeds", "coarse_max_iter", "lm_max_iter"),
    )

    # Warm-up: compile + discard
    for _ in range(N_WARMUP):
        out = jit_solve(
            robot, target_link_index, target_poses[0], rng0, mid_cfg,
            fixed_joint_mask=fixed_joint_mask,
        )
        jax.block_until_ready(out)

    print("Warming up CUDA solver ...")
    for _ in range(N_WARMUP):
        out = hjcd_solve_cuda(
            robot, target_link_index, target_poses[0], rng0, mid_cfg,
            fixed_joint_mask=fixed_joint_mask,
            **IK_KWARGS_CUDA,
        )
        jax.block_until_ready(out)

    # ------------------------------------------------------------------
    # Per-pose evaluation
    # ------------------------------------------------------------------
    print(f"\n{'#':>4}  {'pos JAX (mm)':>13} {'rot JAX (rad)':>13} {'t JAX (ms)':>11}"
          f"  {'pos CUDA (mm)':>13} {'rot CUDA (rad)':>13} {'t CUDA (ms)':>12}"
          f"  {'Δpos (mm)':>9} {'Δori (rad)':>10} {'Δq max':>8}")
    print("-" * 120)

    results_jax:  list[SolveResult] = []
    results_cuda: list[SolveResult] = []

    prev_jax  = mid_cfg
    prev_cuda = mid_cfg

    for i, target_pose in enumerate(target_poses):
        rng_key = jax.random.PRNGKey(i + 1)

        r_jax = _solve_jax(
            robot, target_link_index, target_pose,
            prev_jax, fixed_joint_mask, rng_key, jit_solve, N_TIMED,
        )
        r_cuda = _solve_cuda(
            robot, target_link_index, target_pose,
            prev_cuda, fixed_joint_mask, rng_key, N_TIMED,
        )

        results_jax.append(r_jax)
        results_cuda.append(r_cuda)

        # Cross-solver task-space agreement
        Ts_jax  = robot.forward_kinematics(jnp.array(r_jax.cfg))
        Ts_cuda = robot.forward_kinematics(jnp.array(r_cuda.cfg))
        pose_jax  = jaxlie.SE3(Ts_jax[target_link_index])
        pose_cuda = jaxlie.SE3(Ts_cuda[target_link_index])
        delta_pos = float(jnp.linalg.norm(pose_jax.translation() - pose_cuda.translation())) * 1e3
        delta_ori = float(jnp.linalg.norm(
            (pose_jax.rotation().inverse() @ pose_cuda.rotation()).log()
        ))
        delta_q   = float(np.max(np.abs(r_jax.cfg - r_cuda.cfg)))

        print(f"{i+1:>4}  "
              f"{r_jax.pos_err*1e3:>13.4f} {r_jax.rot_err:>13.6f} {r_jax.time_ms:>11.3f}  "
              f"{r_cuda.pos_err*1e3:>13.4f} {r_cuda.rot_err:>13.6f} {r_cuda.time_ms:>12.3f}  "
              f"{delta_pos:>9.4f} {delta_ori:>10.6f} {delta_q:>8.5f}")

        prev_jax  = jnp.array(r_jax.cfg)
        prev_cuda = jnp.array(r_cuda.cfg)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)

    def _stats(vals: list[float], unit: str = "") -> str:
        a = np.array(vals)
        return (f"mean={a.mean():.4f}{unit}  median={np.median(a):.4f}{unit}"
                f"  p95={np.percentile(a, 95):.4f}{unit}  max={a.max():.4f}{unit}")

    jax_pos  = [r.pos_err * 1e3 for r in results_jax]
    jax_rot  = [r.rot_err       for r in results_jax]
    jax_t    = [r.time_ms       for r in results_jax]
    cuda_pos = [r.pos_err * 1e3 for r in results_cuda]
    cuda_rot = [r.rot_err       for r in results_cuda]
    cuda_t   = [r.time_ms       for r in results_cuda]

    print(f"\n{'':>6}{'Position error (mm)':>32}   {'Rotation error (rad)':>32}   {'Solve time (ms)':>32}")
    print(f"  JAX  : pos  {_stats(jax_pos,  ' mm')}")
    print(f"         rot  {_stats(jax_rot,  ' rad')}")
    print(f"         time {_stats(jax_t,    ' ms')}")
    print(f"  CUDA : pos  {_stats(cuda_pos, ' mm')}")
    print(f"         rot  {_stats(cuda_rot, ' rad')}")
    print(f"         time {_stats(cuda_t,   ' ms')}")

    # Speed comparison
    median_speedup = np.median(jax_t) / np.median(cuda_t)
    print(f"\n  Speedup CUDA vs JAX (median solve time): {median_speedup:.2f}x")

    # Agreement check
    print("\n  Cross-solver agreement (task space):")
    agree_pos_all  = all(
        np.linalg.norm(
            np.array(robot.forward_kinematics(jnp.array(results_jax[i].cfg))[target_link_index, 4:7])
            - np.array(robot.forward_kinematics(jnp.array(results_cuda[i].cfg))[target_link_index, 4:7])
        ) * 1e3 < AGREE_POS_MM
        for i in range(N_TARGETS)
    )
    agree_ori_all = True
    for i in range(N_TARGETS):
        Ts_j = robot.forward_kinematics(jnp.array(results_jax[i].cfg))
        Ts_c = robot.forward_kinematics(jnp.array(results_cuda[i].cfg))
        pj = jaxlie.SE3(Ts_j[target_link_index])
        pc = jaxlie.SE3(Ts_c[target_link_index])
        d_ori = float(jnp.linalg.norm((pj.rotation().inverse() @ pc.rotation()).log()))
        if d_ori >= AGREE_ORI_RAD:
            agree_ori_all = False
            break

    pos_agree_str = f"OK (<{AGREE_POS_MM} mm)"  if agree_pos_all  else f"FAIL (>={AGREE_POS_MM} mm)"
    ori_agree_str = f"OK (<{AGREE_ORI_RAD} rad)" if agree_ori_all else f"FAIL (>={AGREE_ORI_RAD} rad)"
    print(f"    Position : {pos_agree_str}")
    print(f"    Rotation : {ori_agree_str}")

    # Success-rate (pose error < 1 mm and < 0.05 rad = "solved")
    POS_THR_M   = 1e-3   # 1 mm
    ROT_THR_RAD = 0.05   # ~3 deg
    solved_jax  = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results_jax)
    solved_cuda = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results_cuda)
    print(f"\n  Success rate (pos<1mm & rot<0.05rad):")
    print(f"    JAX  : {solved_jax}/{N_TARGETS}  ({100*solved_jax/N_TARGETS:.1f}%)")
    print(f"    CUDA : {solved_cuda}/{N_TARGETS}  ({100*solved_cuda/N_TARGETS:.1f}%)")

    print()


if __name__ == "__main__":
    main()
