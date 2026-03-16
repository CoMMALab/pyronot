"""Benchmark and correctness evaluation: HJCD-IK, LS-IK, and SQP-IK (JAX and CUDA).

Sequential (per-problem) timing
    JAX solvers are evaluated one pose at a time to measure single-problem
    latency.  CUDA solvers are also timed sequentially for a like-for-like
    comparison.

Batch timing
    JAX and CUDA batch solvers are timed over all N_TARGETS at once to measure
    throughput.  The effective per-problem time is batch_wall_time / N_TARGETS.

Correctness
    For each solver the median position / rotation errors across all target
    poses are reported.  JAX vs CUDA agreement is checked at batch level.

Usage:
    python tests/bench_ik.py

Prerequisites:
    1. A CUDA-capable GPU.
    2. CUDA libraries compiled:
           bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_sqp_ik_cuda.sh
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

from pyronot.optimization_engines._hjcd_ik import (
    hjcd_solve,
    hjcd_solve_cuda,
    hjcd_solve_cuda_batch,
)
from pyronot.optimization_engines._ls_ik import (
    ls_ik_solve,
    ls_ik_solve_cuda,
    ls_ik_solve_cuda_batch,
)
from pyronot.optimization_engines._sqp_ik import (
    sqp_ik_solve,
    sqp_ik_solve_cuda,
    sqp_ik_solve_cuda_batch,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAME       = "panda"
TARGET_LINK_NAME = "panda_hand"

# Joints to keep fixed during IK (finger joints for the Panda).
FIXED_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

N_TARGETS = 100    # number of random target poses to evaluate
N_WARMUP  = 3      # JIT / kernel warm-up calls (discarded from timing)
N_TIMED   = 5      # timed repetitions (sequential: per pose; batch: per full call)

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

# LS-IK hyper-parameters.
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

# SQP-IK hyper-parameters.
IK_KWARGS_SQP_JAX = dict(
    num_seeds         = 32,
    max_iter          = 60,
    n_inner_iters     = 2,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)
IK_KWARGS_SQP_CUDA = dict(
    **IK_KWARGS_SQP_JAX,
    eps_pos = 1e-8,
    eps_ori = 1e-8,
)

# Agreement threshold between JAX and CUDA outputs.
AGREE_POS_MM  = 5.0
AGREE_ORI_RAD = 0.5

# Success threshold.
POS_THR_M   = 1e-3
ROT_THR_RAD = 0.05

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    cfg:     np.ndarray
    pos_err: float
    rot_err: float
    time_ms: float   # per-problem time in ms


@dataclass
class BatchResult:
    cfgs:     np.ndarray   # (N_TARGETS, n_act)
    pos_errs: np.ndarray   # (N_TARGETS,) metres
    rot_errs: np.ndarray   # (N_TARGETS,) radians
    time_ms:  float        # effective per-problem time (total / N_TARGETS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_errors(
    robot: pk.Robot,
    cfg: jax.Array,
    target_link_index: int,
    target_pose: jaxlie.SE3,
) -> tuple[float, float]:
    Ts     = robot.forward_kinematics(cfg)
    actual = jaxlie.SE3(Ts[target_link_index])
    pos_err = float(jnp.linalg.norm(actual.translation() - target_pose.translation()))
    rot_err = float(jnp.linalg.norm(
        (target_pose.rotation().inverse() @ actual.rotation()).log()
    ))
    return pos_err, rot_err


def _time_solve(fn, *args, n: int = N_TIMED, **kwargs) -> tuple[jax.Array, float]:
    """Run fn n times, return (last_output, median_wall_time_s)."""
    times = []
    out   = None
    for _ in range(n):
        t0  = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def _run_solver_sequential(
    fn, robot, target_link_index, target_poses, fixed_joint_mask,
    rng_keys, previous_cfgs, kwargs,
) -> list[SolveResult]:
    """Run a single-problem solver sequentially over all target poses."""
    results = []
    for i, target_pose in enumerate(target_poses):
        cfg, t = _time_solve(
            fn, robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_keys[i],
            previous_cfg=previous_cfgs[i],
            fixed_joint_mask=fixed_joint_mask,
            **kwargs,
        )
        pos_err, rot_err = _pose_errors(robot, cfg, target_link_index, target_pose)
        results.append(SolveResult(np.array(cfg), pos_err, rot_err, t * 1e3))
    return results


def _run_solver_batch(
    fn, robot, target_link_index, target_poses_stacked, fixed_joint_mask,
    rng_key, previous_cfgs, kwargs,
) -> BatchResult:
    """Run a batch solver once over all N_TARGETS, time it N_TIMED times."""
    tli = (target_link_index,)
    cfgs_out, total_t = _time_solve(
        fn, robot, tli, target_poses_stacked,
        rng_key, previous_cfgs,
        fixed_joint_mask=fixed_joint_mask,
        **kwargs,
    )
    cfgs_np = np.array(cfgs_out)  # (N_TARGETS, n_act)

    pos_errs = np.empty(len(target_poses_stacked.wxyz_xyz))
    rot_errs = np.empty(len(target_poses_stacked.wxyz_xyz))
    for i in range(len(pos_errs)):
        target_pose = jaxlie.SE3(target_poses_stacked.wxyz_xyz[i])
        pos_errs[i], rot_errs[i] = _pose_errors(
            robot, jnp.array(cfgs_np[i]), target_link_index, target_pose
        )

    effective_ms = total_t * 1e3 / len(pos_errs)
    return BatchResult(cfgs_np, pos_errs, rot_errs, effective_ms)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _stats(vals, unit: str = "") -> str:
    a = np.asarray(vals)
    return (f"mean={a.mean():.4f}{unit}  median={np.median(a):.4f}{unit}"
            f"  p95={np.percentile(a, 95):.4f}{unit}  max={a.max():.4f}{unit}")


def _print_summary_seq(label: str, results: list[SolveResult]) -> None:
    pos    = [r.pos_err * 1e3 for r in results]
    rot    = [r.rot_err       for r in results]
    t      = [r.time_ms       for r in results]
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    print(f"  {label}")
    print(f"    pos  {_stats(pos, ' mm')}")
    print(f"    rot  {_stats(rot, ' rad')}")
    print(f"    time {_stats(t,   ' ms')}  [per-problem sequential]")
    print(f"    success: {solved}/{len(results)}  ({100*solved/len(results):.1f}%)")


def _print_summary_batch(label: str, result: BatchResult) -> None:
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)
    print(f"  {label}")
    print(f"    pos  {_stats(pos, ' mm')}")
    print(f"    rot  {_stats(rot, ' rad')}")
    print(f"    time {result.time_ms:.4f} ms/problem  [batch, N={n}]")
    print(f"    success: {solved}/{n}  ({100*solved/n:.1f}%)")


def _make_batched_jax_solver(base_fn, ik_kwargs):
    """Create a JITted batched JAX solver (vmap over targets)."""
    def _solve_batch(
        robot, target_link_indices, target_poses, rng_keys, previous_cfgs, fixed_joint_mask
    ):
        def _single(target_pose, rng_key, previous_cfg):
            return base_fn(
                robot=robot,
                target_link_indices=target_link_indices,
                target_poses=(target_pose,),
                rng_key=rng_key,
                previous_cfg=previous_cfg,
                fixed_joint_mask=fixed_joint_mask,
                **ik_kwargs,
            )
        return jax.vmap(_single, in_axes=(0, 0, 0))(target_poses, rng_keys, previous_cfgs)

    return jax.jit(_solve_batch, static_argnames=("target_link_indices",))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 80)
    print(f"IK benchmark: HJCD-IK, LS-IK, and SQP-IK  (robot={ROBOT_NAME}, "
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
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)
    rng_np  = np.random.default_rng(0)

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

    # Stack all poses into a single batched SE3 for the batch solvers.
    target_poses_stacked = jaxlie.SE3(
        jnp.stack([p.wxyz_xyz for p in target_poses])
    )  # (N_TARGETS, 7)

    # Per-pose RNG keys and warm-start configs.
    rng_keys     = [jax.random.PRNGKey(i + 1) for i in range(N_TARGETS)]
    rng_keys_batch = jnp.stack(rng_keys)
    previous_cfgs_seq = [mid_cfg] * N_TARGETS  # list for sequential solver
    previous_cfgs_batch = jnp.tile(mid_cfg[None], (N_TARGETS, 1))  # (N_TARGETS, n_act)

    # ------------------------------------------------------------------
    # JIT-compile / warm up all solvers
    # ------------------------------------------------------------------
    rng0 = jax.random.PRNGKey(0)

    jit_hjcd = jax.jit(
        functools.partial(hjcd_solve, **IK_KWARGS_HJCD_JAX),
        static_argnames=("target_link_indices", "num_seeds", "coarse_max_iter", "lm_max_iter"),
    )
    jit_ls = jax.jit(
        functools.partial(ls_ik_solve, **IK_KWARGS_LS_JAX),
        static_argnames=("target_link_indices", "num_seeds", "max_iter"),
    )
    jit_sqp = jax.jit(
        functools.partial(sqp_ik_solve, **IK_KWARGS_SQP_JAX),
        static_argnames=("target_link_indices", "num_seeds", "max_iter", "n_inner_iters"),
    )
    jit_hjcd_batch = _make_batched_jax_solver(hjcd_solve, IK_KWARGS_HJCD_JAX)
    jit_ls_batch   = _make_batched_jax_solver(ls_ik_solve, IK_KWARGS_LS_JAX)
    jit_sqp_batch  = _make_batched_jax_solver(sqp_ik_solve, IK_KWARGS_SQP_JAX)

    warmup_seq = [
        ("HJCD-JAX",  jit_hjcd,        {}),
        ("HJCD-CUDA", hjcd_solve_cuda,  IK_KWARGS_HJCD_CUDA),
        ("LS-JAX",    jit_ls,          {}),
        ("LS-CUDA",   ls_ik_solve_cuda, IK_KWARGS_LS_CUDA),
        ("SQP-JAX",   jit_sqp,         {}),
        ("SQP-CUDA",  sqp_ik_solve_cuda, IK_KWARGS_SQP_CUDA),
    ]
    warmup_batch_jax = [
        ("HJCD-JAX-BATCH", jit_hjcd_batch, {}),
        ("LS-JAX-BATCH",   jit_ls_batch,   {}),
        ("SQP-JAX-BATCH",  jit_sqp_batch,  {}),
    ]
    warmup_batch_cuda = [
        ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,   IK_KWARGS_LS_CUDA),
        ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,     IK_KWARGS_HJCD_CUDA),
        ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch,  IK_KWARGS_SQP_CUDA),
    ]

    tli = (target_link_index,)

    for name, fn, kwargs in warmup_seq:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot=robot, target_link_indices=tli, target_poses=(target_poses[0],),
                     rng_key=rng0, previous_cfg=mid_cfg,
                     fixed_joint_mask=fixed_joint_mask, **kwargs)
            jax.block_until_ready(out)

    for name, fn, kwargs in warmup_batch_jax:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot, tli, target_poses_stacked, rng_keys_batch,
                     previous_cfgs_batch, fixed_joint_mask)
            jax.block_until_ready(out)

    for name, fn, kwargs in warmup_batch_cuda:
        print(f"Warming up {name} ...")
        for _ in range(N_WARMUP):
            out = fn(robot=robot, target_link_indices=tli, target_poses=target_poses_stacked,
                     rng_key=rng0, previous_cfgs=previous_cfgs_batch,
                     fixed_joint_mask=fixed_joint_mask, **kwargs)
            jax.block_until_ready(out)

    # ------------------------------------------------------------------
    # Sequential evaluation (JAX + CUDA single-problem)
    # ------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print("Sequential evaluation (per-problem latency) ...")
    print(f"{'─'*80}")

    seq_solvers = [
        ("HJCD-JAX",  jit_hjcd,         {}),
        ("HJCD-CUDA", hjcd_solve_cuda,   IK_KWARGS_HJCD_CUDA),
        ("LS-JAX",    jit_ls,           {}),
        ("LS-CUDA",   ls_ik_solve_cuda,  IK_KWARGS_LS_CUDA),
        ("SQP-JAX",   jit_sqp,          {}),
        ("SQP-CUDA",  sqp_ik_solve_cuda, IK_KWARGS_SQP_CUDA),
    ]
    seq_results: dict[str, list[SolveResult]] = {}

    for name, fn, kwargs in seq_solvers:
        print(f"  Running {name} ...")
        seq_results[name] = _run_solver_sequential(
            fn, robot, target_link_index, target_poses,
            fixed_joint_mask, rng_keys, previous_cfgs_seq, kwargs,
        )


    # ------------------------------------------------------------------
    # Batch evaluation (JAX + CUDA batch solvers)
    # ------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print("Batch evaluation (all targets in one kernel launch) ...")
    print(f"{'─'*80}")

    batch_solvers = [
        ("LS-JAX-BATCH",    jit_ls_batch,          {},                rng_keys_batch),
        ("HJCD-JAX-BATCH",  jit_hjcd_batch,         {},                rng_keys_batch),
        ("SQP-JAX-BATCH",   jit_sqp_batch,          {},                rng_keys_batch),
        ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,  IK_KWARGS_LS_CUDA,  rng0),
        ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,   IK_KWARGS_HJCD_CUDA, rng0),
        ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch, IK_KWARGS_SQP_CUDA,  rng0),
    ]
    batch_results: dict[str, BatchResult] = {}

    for name, fn, kwargs, rng in batch_solvers:
        print(f"  Running {name} ...")
        batch_results[name] = _run_solver_batch(
            fn, robot, target_link_index, target_poses_stacked,
            fixed_joint_mask, rng, previous_cfgs_batch, kwargs,
        )


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY — Sequential (per-problem latency)")
    print(f"{'='*80}")
    for label in ("HJCD-JAX", "HJCD-CUDA", "LS-JAX", "LS-CUDA", "SQP-JAX", "SQP-CUDA"):
        _print_summary_seq(label, seq_results[label])
        print()

    print(f"{'='*80}")
    print(f"SUMMARY — Batch (effective per-problem time over {N_TARGETS} targets)")
    print(f"{'='*80}")
    for label in ("LS-JAX-BATCH", "HJCD-JAX-BATCH", "SQP-JAX-BATCH",
                  "LS-CUDA-BATCH", "HJCD-CUDA-BATCH", "SQP-CUDA-BATCH"):
        _print_summary_batch(label, batch_results[label])
        print()

    # ------------------------------------------------------------------
    # Speedup table
    # ------------------------------------------------------------------
    print(f"{'='*80}")
    print("Speedups (median sequential time vs effective batch time)")
    print(f"{'='*80}")

    def med_seq(k):
        return float(np.median([r.time_ms for r in seq_results[k]]))

    t = {k: med_seq(k) for k in seq_results}
    t["LS-JAX-BATCH"]    = batch_results["LS-JAX-BATCH"].time_ms
    t["HJCD-JAX-BATCH"]  = batch_results["HJCD-JAX-BATCH"].time_ms
    t["SQP-JAX-BATCH"]   = batch_results["SQP-JAX-BATCH"].time_ms
    t["LS-CUDA-BATCH"]   = batch_results["LS-CUDA-BATCH"].time_ms
    t["HJCD-CUDA-BATCH"] = batch_results["HJCD-CUDA-BATCH"].time_ms
    t["SQP-CUDA-BATCH"]  = batch_results["SQP-CUDA-BATCH"].time_ms

    rows = [
        ("HJCD-CUDA vs HJCD-JAX (sequential)",       "HJCD-JAX",      "HJCD-CUDA"),
        ("LS-CUDA   vs LS-JAX   (sequential)",        "LS-JAX",        "LS-CUDA"),
        ("SQP-CUDA  vs SQP-JAX  (sequential)",        "SQP-JAX",       "SQP-CUDA"),
        ("LS-CUDA-BATCH   vs LS-JAX-BATCH   (batch)", "LS-JAX-BATCH",  "LS-CUDA-BATCH"),
        ("HJCD-CUDA-BATCH vs HJCD-JAX-BATCH (batch)", "HJCD-JAX-BATCH","HJCD-CUDA-BATCH"),
        ("SQP-CUDA-BATCH  vs SQP-JAX-BATCH  (batch)", "SQP-JAX-BATCH", "SQP-CUDA-BATCH"),
        ("LS-CUDA-BATCH   vs LS-JAX   (seq→batch)",   "LS-JAX",        "LS-CUDA-BATCH"),
        ("HJCD-CUDA-BATCH vs HJCD-JAX (seq→batch)",   "HJCD-JAX",      "HJCD-CUDA-BATCH"),
        ("SQP-CUDA-BATCH  vs SQP-JAX  (seq→batch)",   "SQP-JAX",       "SQP-CUDA-BATCH"),
        ("SQP-CUDA vs LS-CUDA (sequential)",           "LS-CUDA",       "SQP-CUDA"),
        ("SQP-CUDA-BATCH vs LS-CUDA-BATCH (batch)",    "LS-CUDA-BATCH", "SQP-CUDA-BATCH"),
    ]
    for desc, slow, fast in rows:
        ratio = t[slow] / t[fast]
        faster = "faster" if ratio >= 1.0 else "slower"
        print(f"  {desc:<55}: {ratio:.2f}x {faster}")

    # ------------------------------------------------------------------
    # JAX vs CUDA-BATCH agreement
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("JAX vs CUDA-BATCH agreement")
    print(f"{'='*80}")

    for jax_key, batch_key in [("HJCD-JAX-BATCH", "HJCD-CUDA-BATCH"),
                               ("LS-JAX-BATCH",   "LS-CUDA-BATCH"),
                               ("SQP-JAX-BATCH",  "SQP-CUDA-BATCH")]:
        delta_pos = []
        delta_ori = []
        for i in range(N_TARGETS):
            Ts_jax  = robot.forward_kinematics(jnp.array(batch_results[jax_key].cfgs[i]))
            Ts_cuda = robot.forward_kinematics(jnp.array(batch_results[batch_key].cfgs[i]))
            pa = jaxlie.SE3(Ts_jax[target_link_index])
            pb = jaxlie.SE3(Ts_cuda[target_link_index])
            delta_pos.append(float(jnp.linalg.norm(pa.translation() - pb.translation())) * 1e3)
            delta_ori.append(float(jnp.linalg.norm((pa.rotation().inverse() @ pb.rotation()).log())))

        pos_ok = all(d < AGREE_POS_MM  for d in delta_pos)
        ori_ok = all(d < AGREE_ORI_RAD for d in delta_ori)
        label  = jax_key.split("-")[0]
        print(f"  {label:<6}: pos={'OK' if pos_ok else 'FAIL'}  "
              f"(mean {np.mean(delta_pos):.3f} mm, max {np.max(delta_pos):.3f} mm)  "
              f"ori={'OK' if ori_ok else 'FAIL'}  "
              f"(mean {np.mean(delta_ori):.4f} rad, max {np.max(delta_ori):.4f} rad)")

    # ------------------------------------------------------------------
    # SQP vs LS accuracy comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SQP vs LS accuracy comparison (batch)")
    print(f"{'='*80}")

    sqp_pos  = batch_results["SQP-CUDA-BATCH"].pos_errs * 1e3
    ls_pos   = batch_results["LS-CUDA-BATCH"].pos_errs  * 1e3
    sqp_succ = int(np.sum(
        (batch_results["SQP-CUDA-BATCH"].pos_errs < POS_THR_M) &
        (batch_results["SQP-CUDA-BATCH"].rot_errs < ROT_THR_RAD)
    ))
    ls_succ  = int(np.sum(
        (batch_results["LS-CUDA-BATCH"].pos_errs  < POS_THR_M) &
        (batch_results["LS-CUDA-BATCH"].rot_errs  < ROT_THR_RAD)
    ))
    print(f"  SQP-CUDA-BATCH  median pos {np.median(sqp_pos):.4f} mm  success {sqp_succ}/{N_TARGETS}")
    print(f"  LS-CUDA-BATCH   median pos {np.median(ls_pos):.4f} mm  success {ls_succ}/{N_TARGETS}")
    print()


if __name__ == "__main__":
    main()
