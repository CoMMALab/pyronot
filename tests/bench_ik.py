"""Benchmark and correctness evaluation: HJCD-IK and LS-IK solvers (JAX and CUDA).

Measures for each solver across a set of randomised target poses:
  - Median solve time (ms)
  - Position error (mm)  — ‖t_actual − t_target‖₂
  - Rotation error (rad) — ‖log(R_target⁻¹ R_actual)‖₂
  - Agreement between JAX and CUDA variants of each solver

Usage:
    python tests/bench_ik.py

Prerequisites:
    1. A CUDA-capable GPU.
    2. CUDA libraries compiled:
           bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh
    3. robot_descriptions installed:
           pip install robot_descriptions
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from pyronot.optimization_engines._hjcd_ik import hjcd_solve, hjcd_solve_cuda
from pyronot.optimization_engines._ls_ik import ls_ik_solve, ls_ik_solve_cuda

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAME       = "panda"
TARGET_LINK_NAME = "panda_hand"

# Joints to keep fixed during IK (finger joints for the Panda).
FIXED_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

N_TARGETS = 2000   # number of random target poses to evaluate
N_WARMUP  = 3      # JIT / kernel warm-up calls (discarded from timing)
N_TIMED   = 20     # timed repetitions per pose (median reported)

# HJCD-IK hyper-parameters.
IK_KWARGS_HJCD_JAX = dict(
    num_seeds          = 32,
    coarse_max_iter    = 20,
    lm_max_iter        = 40,
    lambda_init        = 1e-3,
    continuity_weight  = 0.0,
    limit_prior_weight = 1e-4,
    kick_scale         = 0.02,
)
IK_KWARGS_HJCD_CUDA = dict(**IK_KWARGS_HJCD_JAX)

# LS-IK hyper-parameters (no coarse phase, fixed pos/ori weights).
IK_KWARGS_LS_JAX = dict(
    num_seeds         = 32,
    max_iter          = 60,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)
IK_KWARGS_LS_CUDA = dict(
    **IK_KWARGS_LS_JAX,
    eps_pos = 1e-8,
    eps_ori = 1e-8,
)

# Agreement threshold between JAX and CUDA outputs (per solver).
AGREE_POS_MM  = 5.0   # mm
AGREE_ORI_RAD = 0.5   # rad

# Success threshold ("solved").
POS_THR_M   = 1e-3   # 1 mm
ROT_THR_RAD = 0.05   # ~3 deg

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    cfg:     np.ndarray   # (n_act,)
    pos_err: float        # metres
    rot_err: float        # radians
    time_ms: float        # median solve time (milliseconds)


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


def _run_solver(fn, robot, target_link_index, target_pose,
                previous_cfg, fixed_joint_mask, rng_key, kwargs, n_timed) -> SolveResult:
    cfg, t = _time_solve(
        fn, robot, target_link_index, target_pose, rng_key, previous_cfg,
        fixed_joint_mask=fixed_joint_mask,
        n=n_timed,
        **kwargs,
    )
    pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)
    return SolveResult(np.array(cfg), pos_err, rot_err, t * 1e3)


def _cross_agreement(
    robot: pk.Robot,
    target_link_index: int,
    r_a: SolveResult,
    r_b: SolveResult,
) -> tuple[float, float, float]:
    """Returns (delta_pos_mm, delta_ori_rad, delta_q_max)."""
    Ts_a = robot.forward_kinematics(jnp.array(r_a.cfg))
    Ts_b = robot.forward_kinematics(jnp.array(r_b.cfg))
    pa = jaxlie.SE3(Ts_a[target_link_index])
    pb = jaxlie.SE3(Ts_b[target_link_index])
    delta_pos = float(jnp.linalg.norm(pa.translation() - pb.translation())) * 1e3
    delta_ori = float(jnp.linalg.norm((pa.rotation().inverse() @ pb.rotation()).log()))
    delta_q   = float(np.max(np.abs(r_a.cfg - r_b.cfg)))
    return delta_pos, delta_ori, delta_q


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _stats(vals: list[float], unit: str = "") -> str:
    a = np.array(vals)
    return (f"mean={a.mean():.4f}{unit}  median={np.median(a):.4f}{unit}"
            f"  p95={np.percentile(a, 95):.4f}{unit}  max={a.max():.4f}{unit}")


def _print_summary_block(label: str, results: list[SolveResult]) -> None:
    pos  = [r.pos_err * 1e3 for r in results]
    rot  = [r.rot_err       for r in results]
    t    = [r.time_ms       for r in results]
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    print(f"  {label}")
    print(f"    pos  {_stats(pos, ' mm')}")
    print(f"    rot  {_stats(rot, ' rad')}")
    print(f"    time {_stats(t,   ' ms')}")
    print(f"    success (pos<1mm & rot<0.05rad): {solved}/{len(results)}"
          f"  ({100*solved/len(results):.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 80)
    print(f"IK benchmark: HJCD-IK and LS-IK  (robot={ROBOT_NAME}, "
          f"n_targets={N_TARGETS}, n_timed={N_TIMED})")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load robot
    # ------------------------------------------------------------------
    print("\nLoading robot ...")
    urdf  = load_robot_description(f"{ROBOT_NAME}_description")
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints
    target_link_index = robot.links.names.index(TARGET_LINK_NAME)

    fixed_joint_mask = jnp.array(
        [name in FIXED_JOINT_NAMES for name in robot.joints.actuated_names],
        dtype=jnp.int32,
    )
    print(f"  {n_act} actuated joints, target link: '{TARGET_LINK_NAME}'")
    print(f"  Fixed joints: {[n for n in FIXED_JOINT_NAMES if n in robot.joints.actuated_names]}")

    lo     = np.array(robot.joints.lower_limits)
    hi     = np.array(robot.joints.upper_limits)
    rng_np = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Generate target poses
    # ------------------------------------------------------------------
    print("\nGenerating target poses ...")
    target_cfgs_np = rng_np.uniform(lo, hi, size=(N_TARGETS, n_act)).astype(np.float32)
    target_poses: list[jaxlie.SE3] = []
    for i in range(N_TARGETS):
        cfg_i = jnp.array(target_cfgs_np[i])
        Ts    = robot.forward_kinematics(cfg_i)
        target_poses.append(jaxlie.SE3(Ts[target_link_index]))

    # ------------------------------------------------------------------
    # JIT-compile / warm up all solvers
    # ------------------------------------------------------------------
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)
    rng0    = jax.random.PRNGKey(0)

    jit_hjcd = jax.jit(
        functools.partial(hjcd_solve, **IK_KWARGS_HJCD_JAX),
        static_argnames=("target_link_index", "num_seeds", "coarse_max_iter", "lm_max_iter"),
    )
    jit_ls = jax.jit(
        functools.partial(ls_ik_solve, **IK_KWARGS_LS_JAX),
        static_argnames=("target_link_index", "num_seeds", "max_iter"),
    )

    warmup_solvers = [
        ("HJCD-JAX",  jit_hjcd,        {}),
        ("HJCD-CUDA", hjcd_solve_cuda,  IK_KWARGS_HJCD_CUDA),
        ("LS-JAX",    jit_ls,          {}),
        ("LS-CUDA",   ls_ik_solve_cuda, IK_KWARGS_LS_CUDA),
    ]
    for name, fn, kwargs in warmup_solvers:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot, target_link_index, target_poses[0], rng0, mid_cfg,
                     fixed_joint_mask=fixed_joint_mask, **kwargs)
            jax.block_until_ready(out)

    # ------------------------------------------------------------------
    # Per-pose evaluation
    # ------------------------------------------------------------------
    HDR = (f"{'#':>4}  {'Solver':<10}  {'pos (mm)':>10} {'rot (rad)':>10} {'t (ms)':>9}"
           f"   {'Δpos (mm)':>10} {'Δori (rad)':>10} {'Δq max':>8}")
    SEP = "-" * len(HDR)
    print(f"\n{HDR}")
    print(SEP)

    results: dict[str, list[SolveResult]] = {
        "HJCD-JAX":  [],
        "HJCD-CUDA": [],
        "LS-JAX":    [],
        "LS-CUDA":   [],
    }

    prev: dict[str, jax.Array] = {k: mid_cfg for k in results}

    for i, target_pose in enumerate(target_poses):
        rng_key = jax.random.PRNGKey(i + 1)

        r_hjcd_jax = _run_solver(
            jit_hjcd, robot, target_link_index, target_pose,
            prev["HJCD-JAX"], fixed_joint_mask, rng_key, {}, N_TIMED,
        )
        r_hjcd_cuda = _run_solver(
            hjcd_solve_cuda, robot, target_link_index, target_pose,
            prev["HJCD-CUDA"], fixed_joint_mask, rng_key, IK_KWARGS_HJCD_CUDA, N_TIMED,
        )
        r_ls_jax = _run_solver(
            jit_ls, robot, target_link_index, target_pose,
            prev["LS-JAX"], fixed_joint_mask, rng_key, {}, N_TIMED,
        )
        r_ls_cuda = _run_solver(
            ls_ik_solve_cuda, robot, target_link_index, target_pose,
            prev["LS-CUDA"], fixed_joint_mask, rng_key, IK_KWARGS_LS_CUDA, N_TIMED,
        )

        results["HJCD-JAX"].append(r_hjcd_jax)
        results["HJCD-CUDA"].append(r_hjcd_cuda)
        results["LS-JAX"].append(r_ls_jax)
        results["LS-CUDA"].append(r_ls_cuda)

        d_hjcd = _cross_agreement(robot, target_link_index, r_hjcd_jax, r_hjcd_cuda)
        d_ls   = _cross_agreement(robot, target_link_index, r_ls_jax,   r_ls_cuda)

        idx_str = f"{i+1:>4}"
        for label, r, d in [
            ("HJCD-JAX",  r_hjcd_jax,  d_hjcd),
            ("HJCD-CUDA", r_hjcd_cuda, d_hjcd),
            ("LS-JAX",    r_ls_jax,    d_ls),
            ("LS-CUDA",   r_ls_cuda,   d_ls),
        ]:
            agreement = f"{d[0]:>10.4f} {d[1]:>10.6f} {d[2]:>8.5f}" if label.endswith("CUDA") else " " * 32
            print(f"{idx_str}  {label:<10}  "
                  f"{r.pos_err*1e3:>10.4f} {r.rot_err:>10.6f} {r.time_ms:>9.3f}"
                  f"   {agreement}")
            idx_str = "    "  # blank index for continuation lines

        prev["HJCD-JAX"]  = jnp.array(r_hjcd_jax.cfg)
        prev["HJCD-CUDA"] = jnp.array(r_hjcd_cuda.cfg)
        prev["LS-JAX"]    = jnp.array(r_ls_jax.cfg)
        prev["LS-CUDA"]   = jnp.array(r_ls_cuda.cfg)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label in ("HJCD-JAX", "HJCD-CUDA", "LS-JAX", "LS-CUDA"):
        _print_summary_block(label, results[label])
        print()

    # Speed comparisons
    print("  Speedup (median solve time):")
    t = {k: np.median([r.time_ms for r in v]) for k, v in results.items()}
    print(f"    HJCD-CUDA vs HJCD-JAX : {t['HJCD-JAX']  / t['HJCD-CUDA']:.2f}x")
    print(f"    LS-CUDA   vs LS-JAX   : {t['LS-JAX']    / t['LS-CUDA']:.2f}x")
    print(f"    LS-JAX    vs HJCD-JAX : {t['HJCD-JAX']  / t['LS-JAX']:.2f}x")
    print(f"    LS-CUDA   vs HJCD-CUDA: {t['HJCD-CUDA'] / t['LS-CUDA']:.2f}x")

    # Cross-solver agreement (JAX vs CUDA, per solver family)
    print("\n  JAX vs CUDA agreement:")
    for jax_key, cuda_key in [("HJCD-JAX", "HJCD-CUDA"), ("LS-JAX", "LS-CUDA")]:
        pos_ok = ori_ok = True
        for j in range(N_TARGETS):
            dp, do, _ = _cross_agreement(
                robot, target_link_index, results[jax_key][j], results[cuda_key][j]
            )
            if dp >= AGREE_POS_MM:
                pos_ok = False
            if do >= AGREE_ORI_RAD:
                ori_ok = False
        label = jax_key.split("-")[0]
        pos_str = f"OK (<{AGREE_POS_MM} mm)"   if pos_ok else f"FAIL (>={AGREE_POS_MM} mm)"
        ori_str = f"OK (<{AGREE_ORI_RAD} rad)" if ori_ok else f"FAIL (>={AGREE_ORI_RAD} rad)"
        print(f"    {label:<6}: pos={pos_str}  ori={ori_str}")

    print()


if __name__ == "__main__":
    main()
