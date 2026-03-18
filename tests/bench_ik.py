"""Benchmark and correctness evaluation: HJCD-IK, LS-IK, SQP-IK, MPPI-IK, and Learned-IK.

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

Learned-IK
    The Learned-IK solver requires a pre-trained Flax model.  Train one with:
        python train_learned_ik.py --robot panda
    The model is saved to resources/learned_ik/panda.pkl and loaded
    automatically.  If no model is found the learned-IK rows are skipped.

Usage:
    python tests/bench_ik.py

Prerequisites:
    1. A CUDA-capable GPU.
    2. CUDA libraries compiled:
           bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_sqp_ik_cuda.sh
           bash src/pyronot/cuda_kernels/build_mppi_ik_cuda.sh
    3. robot_descriptions installed:
           pip install robot_descriptions
    4. (Optional) Flax model for Learned-IK:
           pip install flax optax
           python train_learned_ik.py --robot panda
"""

from __future__ import annotations

import contextlib
import functools
import threading
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

# Optional NVML for GPU monitoring (nvidia-ml-py / pynvml).
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_HANDLE: object | None = _pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False


@contextlib.contextmanager
def _gpu_monitor(interval_s: float = 0.02):
    """Sample GPU utilisation and VRAM in a background thread.

    Yields a dict; on exit the dict contains:
        ``gpu_util``  – list of utilisation % samples
        ``vram_mb``   – list of used VRAM (MiB) samples
    """
    samples: dict[str, list[float]] = {"gpu_util": [], "vram_mb": []}
    stop_evt = threading.Event()

    def _sample() -> None:
        while not stop_evt.is_set():
            if _NVML_OK and _NVML_HANDLE is not None:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem  = _pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                samples["gpu_util"].append(float(util.gpu))
                samples["vram_mb"].append(float(mem.used) / 1024 ** 2)
            stop_evt.wait(interval_s)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_evt.set()
        t.join(timeout=1.0)

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
from pyronot.optimization_engines._mppi_ik import (
    mppi_ik_solve,
    mppi_ik_solve_cuda,
    mppi_ik_solve_cuda_batch,
)

# Learned-IK: imports only; model is loaded inside main() after the robot is known.
try:
    from pyronot.optimization_engines._learned_ik import (
        get_default_model_path,
        load_learned_ik,
        make_learned_ik_solve,
    )
    _LEARNED_IK_IMPORT_OK = True
except Exception:
    _LEARNED_IK_IMPORT_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAME       = "panda"
TARGET_LINK_NAME = "panda_hand"

# Joints to keep fixed during IK (finger joints for the Panda).
FIXED_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

N_TARGETS = 10    # number of random target poses to evaluate
N_TARGETS_BATCH = 2048
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

# MPPI-IK hyper-parameters.
IK_KWARGS_MPPI_JAX = dict(
    num_seeds         = 32,
    n_particles       = 16,
    n_mppi_iters      = 5,
    n_lbfgs_iters     = 30,
    m_lbfgs           = 5,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    sigma             = 0.3,
    mppi_temperature  = 0.05,
    continuity_weight = 0.0,
)
IK_KWARGS_MPPI_CUDA = dict(
    num_seeds         = 32,
    n_particles       = 16,
    n_mppi_iters      = 5,
    n_lbfgs_iters     = 25,
    m_lbfgs           = 5,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    sigma             = 0.3,
    mppi_temperature  = 0.05,
    eps_pos           = 1e-8,
    eps_ori           = 1e-8,
    continuity_weight = 0.0,
)

# Learned-IK hyper-parameters.
# num_seeds:      half come from the MLP prediction ± noise; half are random.
# n_refine_iters: LM steps run on each seed after the MLP warm-start.
IK_KWARGS_LEARNED_JAX = dict(
    num_seeds         = 64,
    n_refine_iters    = 15,
    pos_weight        = 50.0,
    ori_weight        = 10.0,
    lambda_init       = 5e-3,
    continuity_weight = 0.0,
)

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
    cfgs:          np.ndarray   # (N_TARGETS, n_act)
    pos_errs:      np.ndarray   # (N_TARGETS,) metres
    rot_errs:      np.ndarray   # (N_TARGETS,) radians
    time_ms:       float        # effective per-problem time (total / N_TARGETS)
    peak_gpu_util: float = float("nan")   # peak GPU utilisation (%)
    avg_gpu_util:  float = float("nan")   # mean GPU utilisation (%)
    peak_vram_mb:  float = float("nan")   # peak VRAM used (MiB)


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
    with _gpu_monitor() as gpu_samples:
        cfgs_out, total_t = _time_solve(
            fn, robot, tli, target_poses_stacked,
            rng_key, previous_cfgs,
            fixed_joint_mask=fixed_joint_mask,
            **kwargs,
        )
    cfgs_np = np.array(cfgs_out)  # (N_TARGETS, n_act)

    peak_gpu = max(gpu_samples["gpu_util"], default=float("nan"))
    avg_gpu  = float(np.mean(gpu_samples["gpu_util"])) if gpu_samples["gpu_util"] else float("nan")
    peak_vram = max(gpu_samples["vram_mb"], default=float("nan"))

    pos_errs = np.empty(len(target_poses_stacked.wxyz_xyz))
    rot_errs = np.empty(len(target_poses_stacked.wxyz_xyz))
    for i in range(len(pos_errs)):
        target_pose = jaxlie.SE3(target_poses_stacked.wxyz_xyz[i])
        pos_errs[i], rot_errs[i] = _pose_errors(
            robot, jnp.array(cfgs_np[i]), target_link_index, target_pose
        )

    effective_ms = total_t * 1e3 / len(pos_errs)
    return BatchResult(cfgs_np, pos_errs, rot_errs, effective_ms,
                       peak_gpu_util=peak_gpu, avg_gpu_util=avg_gpu, peak_vram_mb=peak_vram)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_COL_W = 20  # method name column width
_NUM_W = 10  # numeric column width

def _table_header(cols: list[str]) -> str:
    row = f"  {'Method':<{_COL_W}}"
    for c in cols:
        row += f"  {c:>{_NUM_W}}"
    return row

def _table_sep(n_cols: int) -> str:
    return "  " + "-" * (_COL_W + n_cols * (_NUM_W + 2))

def _table_row(label: str, vals: list[str]) -> str:
    row = f"  {label:<{_COL_W}}"
    for v in vals:
        row += f"  {v:>{_NUM_W}}"
    return row


def _seq_row(label: str, results: list[SolveResult]) -> tuple[str, dict]:
    pos    = np.array([r.pos_err * 1e3 for r in results])
    rot    = np.array([r.rot_err       for r in results])
    t      = np.array([r.time_ms       for r in results])
    solved = sum(r.pos_err < POS_THR_M and r.rot_err < ROT_THR_RAD for r in results)
    n      = len(results)
    vals = [
        f"{np.median(t):.3f}",
        f"{np.percentile(t, 95):.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
    ]
    return _table_row(label, vals), {"t_med": float(np.median(t))}


def _batch_row(label: str, result: BatchResult) -> tuple[str, dict]:
    pos    = result.pos_errs * 1e3
    rot    = result.rot_errs
    solved = int(np.sum((result.pos_errs < POS_THR_M) & (result.rot_errs < ROT_THR_RAD)))
    n      = len(pos)

    def _fmt_pct(v: float) -> str:
        return f"{v:.0f}%" if not np.isnan(v) else "n/a"

    def _fmt_mb(v: float) -> str:
        return f"{v:.0f}" if not np.isnan(v) else "n/a"

    vals = [
        f"{result.time_ms:.3f}",
        f"{np.median(pos):.4f}",
        f"{np.percentile(pos, 95):.4f}",
        f"{np.median(rot):.4f}",
        f"{np.percentile(rot, 95):.4f}",
        f"{solved}/{n}",
        _fmt_pct(result.peak_gpu_util),
        _fmt_pct(result.avg_gpu_util),
        _fmt_mb(result.peak_vram_mb),
    ]
    return _table_row(label, vals), {}


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
    print(f"IK benchmark: HJCD-IK, LS-IK, SQP-IK, and MPPI-IK  (robot={ROBOT_NAME}, "
          f"n_targets={N_TARGETS}, n_timed={N_TIMED})")
    gpu_mon_status = "enabled (pynvml)" if _NVML_OK else "disabled (install nvidia-ml-py for GPU stats)"
    print(f"GPU monitoring: {gpu_mon_status}")
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

    # ------------------------------------------------------------------
    # Learned-IK: load pre-trained Flax model (optional)
    # ------------------------------------------------------------------
    _learned_ik_available = False
    _learned_ik_fn        = None
    _learned_ik_fn_batch  = None
    if _LEARNED_IK_IMPORT_OK:
        _model_path = get_default_model_path(ROBOT_NAME)
        if _model_path.exists():
            print(f"\nLoaded Learned-IK model: {_model_path}")
            _model_data  = load_learned_ik(_model_path)
            _model_params = _model_data["params"]
            def _infer_ikflow_arch(params):
                if isinstance(params, dict) and "params" in params:
                    params = params["params"]
                if not isinstance(params, dict):
                    return 15, 1024
                nets = [k for k in params.keys() if k.startswith("nets_")]
                if not nets:
                    return 15, 1024
                n_layers = len(nets)
                hidden = 1024
                try:
                    hidden = int(params[nets[0]]["Dense_0"]["kernel"].shape[1])
                except Exception:
                    pass
                return n_layers, hidden

            _n_layers, _hidden = _infer_ikflow_arch(_model_params)
            _learned_base = make_learned_ik_solve(
                robot,
                latent_dim=_model_data.get("latent_dim", 15),
                n_layers=_n_layers,
                hidden=_hidden,
            )

            # Wrap so that model_params is baked in and the signature
            # matches the other single-problem solvers used in the benchmark.
            def _learned_ik_fn(
                robot, target_link_indices, target_poses,
                rng_key, previous_cfg, fixed_joint_mask=None, **kwargs,
            ):
                return _learned_base(
                    robot, target_link_indices, target_poses,
                    rng_key, previous_cfg,
                    model_params=_model_params,
                    fixed_joint_mask=fixed_joint_mask,
                    **kwargs,
                )

            _learned_ik_fn_batch = _make_batched_jax_solver(
                _learned_ik_fn, IK_KWARGS_LEARNED_JAX,
            )
            _learned_ik_available = True
        else:
            print(f"\nLearned-IK model not found at {_model_path}")
            print("  Run: python train_learned_ik.py --robot panda")
            print("  Learned-IK rows will be skipped in the benchmark.")
    else:
        print("\nLearned-IK unavailable (flax not installed).")

    lo     = np.array(robot.joints.lower_limits)
    hi     = np.array(robot.joints.upper_limits)
    mid_cfg = jnp.array((lo + hi) / 2, dtype=jnp.float32)
    rng_np  = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Generate target poses  (batch superset; sequential uses first N_TARGETS)
    # ------------------------------------------------------------------
    print(f"\nGenerating target poses (seq={N_TARGETS}, batch={N_TARGETS_BATCH}) ...")
    target_cfgs_np = rng_np.uniform(lo, hi, size=(N_TARGETS_BATCH, n_act)).astype(np.float32)
    all_target_poses: list[jaxlie.SE3] = []
    for i in range(N_TARGETS_BATCH):
        cfg_i = jnp.array(target_cfgs_np[i])
        Ts    = robot.forward_kinematics(cfg_i)
        all_target_poses.append(jaxlie.SE3(Ts[target_link_index]))

    # Sequential subset.
    target_poses = all_target_poses[:N_TARGETS]

    # Stack all batch poses into a single batched SE3 for the batch solvers.
    target_poses_stacked = jaxlie.SE3(
        jnp.stack([p.wxyz_xyz for p in all_target_poses])
    )  # (N_TARGETS_BATCH, 7)

    # Per-pose RNG keys and warm-start configs.
    rng_keys          = [jax.random.PRNGKey(i + 1) for i in range(N_TARGETS)]
    rng_keys_batch    = jnp.stack([jax.random.PRNGKey(i + 1) for i in range(N_TARGETS_BATCH)])
    previous_cfgs_seq   = [mid_cfg] * N_TARGETS
    previous_cfgs_batch = jnp.tile(mid_cfg[None], (N_TARGETS_BATCH, 1))  # (N_TARGETS_BATCH, n_act)

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
    jit_mppi = jax.jit(
        functools.partial(mppi_ik_solve, **IK_KWARGS_MPPI_JAX),
        static_argnames=("target_link_indices", "num_seeds", "n_particles",
                         "n_mppi_iters", "n_lbfgs_iters", "m_lbfgs"),
    )
    jit_mppi_batch = _make_batched_jax_solver(mppi_ik_solve, IK_KWARGS_MPPI_JAX)

    warmup_seq = [
        ("HJCD-JAX",   jit_hjcd,          {}),
        ("HJCD-CUDA",  hjcd_solve_cuda,    IK_KWARGS_HJCD_CUDA),
        ("LS-JAX",     jit_ls,            {}),
        ("LS-CUDA",    ls_ik_solve_cuda,   IK_KWARGS_LS_CUDA),
        ("SQP-JAX",    jit_sqp,           {}),
        ("SQP-CUDA",   sqp_ik_solve_cuda,  IK_KWARGS_SQP_CUDA),
        ("MPPI-JAX",   jit_mppi,          {}),
        ("MPPI-CUDA",  mppi_ik_solve_cuda, IK_KWARGS_MPPI_CUDA),
    ]
    if _learned_ik_available:
        warmup_seq.append(("Learned-JAX", _learned_ik_fn, IK_KWARGS_LEARNED_JAX))

    warmup_batch_jax = [
        ("HJCD-JAX-BATCH",  jit_hjcd_batch,  {}),
        ("LS-JAX-BATCH",    jit_ls_batch,    {}),
        ("SQP-JAX-BATCH",   jit_sqp_batch,   {}),
        ("MPPI-JAX-BATCH",  jit_mppi_batch,  {}),
    ]
    if _learned_ik_available:
        warmup_batch_jax.append(("Learned-JAX-BATCH", _learned_ik_fn_batch, {}))
    warmup_batch_cuda = [
        ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,   IK_KWARGS_LS_CUDA),
        ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,     IK_KWARGS_HJCD_CUDA),
        ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch,  IK_KWARGS_SQP_CUDA),
        ("MPPI-CUDA-BATCH", mppi_ik_solve_cuda_batch, IK_KWARGS_MPPI_CUDA),
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
        ("HJCD-JAX",  jit_hjcd,           {}),
        ("HJCD-CUDA", hjcd_solve_cuda,     IK_KWARGS_HJCD_CUDA),
        ("LS-JAX",    jit_ls,             {}),
        ("LS-CUDA",   ls_ik_solve_cuda,    IK_KWARGS_LS_CUDA),
        ("SQP-JAX",   jit_sqp,            {}),
        ("SQP-CUDA",  sqp_ik_solve_cuda,   IK_KWARGS_SQP_CUDA),
        ("MPPI-JAX",  jit_mppi,           {}),
        ("MPPI-CUDA", mppi_ik_solve_cuda,  IK_KWARGS_MPPI_CUDA),
    ]
    if _learned_ik_available:
        seq_solvers.append(("Learned-JAX", _learned_ik_fn, IK_KWARGS_LEARNED_JAX))

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
        ("LS-JAX-BATCH",    jit_ls_batch,             {},                 rng_keys_batch),
        ("HJCD-JAX-BATCH",  jit_hjcd_batch,            {},                 rng_keys_batch),
        ("SQP-JAX-BATCH",   jit_sqp_batch,             {},                 rng_keys_batch),
        ("MPPI-JAX-BATCH",  jit_mppi_batch,            {},                 rng_keys_batch),
        ("LS-CUDA-BATCH",   ls_ik_solve_cuda_batch,    IK_KWARGS_LS_CUDA,   rng0),
        ("HJCD-CUDA-BATCH", hjcd_solve_cuda_batch,      IK_KWARGS_HJCD_CUDA, rng0),
        ("SQP-CUDA-BATCH",  sqp_ik_solve_cuda_batch,   IK_KWARGS_SQP_CUDA,  rng0),
        ("MPPI-CUDA-BATCH", mppi_ik_solve_cuda_batch,  IK_KWARGS_MPPI_CUDA, rng0),
    ]
    if _learned_ik_available:
        batch_solvers.append(("Learned-JAX-BATCH", _learned_ik_fn_batch, {}, rng_keys_batch))

    batch_results: dict[str, BatchResult] = {}

    for name, fn, kwargs, rng in batch_solvers:
        print(f"  Running {name} ...")
        batch_results[name] = _run_solver_batch(
            fn, robot, target_link_index, target_poses_stacked,
            fixed_joint_mask, rng, previous_cfgs_batch, kwargs,
        )


    # ------------------------------------------------------------------
    # Results tables
    # ------------------------------------------------------------------
    seq_cols   = ["t_med(ms)", "t_p95(ms)", "pos_med(mm)", "pos_p95(mm)",
                  "rot_med(rad)", "rot_p95(rad)", "success"]
    batch_cols = ["ms/prob",   "pos_med(mm)", "pos_p95(mm)",
                  "rot_med(rad)", "rot_p95(rad)", "success",
                  "gpu_pk(%)", "gpu_avg(%)", "vram_pk(MB)"]

    seq_order = [
        "HJCD-JAX", "HJCD-CUDA",
        "LS-JAX",   "LS-CUDA",
        "SQP-JAX",  "SQP-CUDA",
        "MPPI-JAX", "MPPI-CUDA",
    ]
    if _learned_ik_available:
        seq_order.append("Learned-JAX")

    batch_order = [
        "HJCD-JAX-BATCH",  "HJCD-CUDA-BATCH",
        "LS-JAX-BATCH",    "LS-CUDA-BATCH",
        "SQP-JAX-BATCH",   "SQP-CUDA-BATCH",
        "MPPI-JAX-BATCH",  "MPPI-CUDA-BATCH",
    ]
    if _learned_ik_available:
        batch_order.append("Learned-JAX-BATCH")

    print(f"\n{'='*80}")
    print(f"Sequential results — per-problem latency  (N={N_TARGETS}, timed={N_TIMED})")
    print(f"{'='*80}")
    print(_table_header(seq_cols))
    print(_table_sep(len(seq_cols)))
    for label in seq_order:
        row, _ = _seq_row(label, seq_results[label])
        print(row)

    print(f"\n{'='*80}")
    print(f"Batch results — effective per-problem time  (N={N_TARGETS_BATCH}, timed={N_TIMED})")
    print(f"{'='*80}")
    print(_table_header(batch_cols))
    print(_table_sep(len(batch_cols)))
    for label in batch_order:
        row, _ = _batch_row(label, batch_results[label])
        print(row)
    print()


if __name__ == "__main__":
    main()
