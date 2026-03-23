"""Benchmark: collision-checking backends for pyronot.

Compares four backends across two operations and several batch sizes:

  Backends:
    JAX-Capsule        — RobotCollision  (one capsule per link, JAX)
    JAX-Sphere         — RobotCollisionSpherized (multi-sphere per link, JAX)
    JAX-Neural         — NeuralRobotCollision trained on the scene
    CUDA-Capsule       — CUDARobotCollisionChecker wrapping RobotCollision
    CUDA-Sphere        — CUDARobotCollisionChecker wrapping RobotCollisionSpherized
    CUDA-Sphere-Coarse — CUDARobotCollisionChecker with coarse-first guard:
                         runs the coarse (1-sphere/link) model first; if the
                         coarse check is collision-free the fine kernel is
                         skipped and all-+1 distances are returned.
                         NOT differentiable — do not use in trajopt.

  Operations:
    compute_world_collision_distance(robot, cfg, world_geom)
    compute_self_collision_distance(robot, cfg)

  Metrics (per backend × batch size):
    ms/call  — wall-clock milliseconds for the full batched call
    ms/cfg   — effective per-configuration time  (ms/call / batch_size)
    gpu_pk   — peak GPU utilisation %  (if pynvml available)
    vram_pk  — peak VRAM used (MiB)    (if pynvml available)

Usage:
    python tests/bench_collision.py [--robot ROBOT] [--neural-samples N]

Prerequisites:
    pip install robot_descriptions
    bash src/pyronot/cuda_kernels/build_collision_cuda.sh  (for CUDA backends)
    (pynvml optional, for GPU monitoring: pip install nvidia-ml-py)

Neural training:
    The neural SDF model is trained in-process before benchmarking.
    Use --neural-samples to control training set size (default 4000).
    Pass --skip-neural to skip neural training and benchmarking.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import threading
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import yourdfpy
import pyronot as pk
from loguru import logger
from robot_descriptions.loaders.yourdfpy import load_robot_description

from pyronot.collision import (
    CUDARobotCollisionChecker,
    NeuralRobotCollision,
    RobotCollision,
    RobotCollisionSpherized,
    Sphere,
    Box,
    Capsule,
    HalfSpace,
)

# Optional GPU monitoring
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_HANDLE: object | None = _pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception:
    _NVML_HANDLE = None
    _NVML_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_NAME = "panda"

# Local URDFs for spherized collision models (have sphere primitives, not meshes)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SPHERIZED_URDF       = _REPO_ROOT / "resources" / "panda" / "panda_spherized.urdf"
COARSE_SPHERIZED_URDF = _REPO_ROOT / "resources" / "panda" / "panda_spherized_coarse.urdf"

# Number of random configs to use as the test set
N_WARMUP = 3          # JIT / kernel warm-up calls (results discarded)
N_TIMED  = 7          # timed repetitions; median is reported

# Batch sizes to sweep
BATCH_SIZES = [1, 64, 512, 2048]

# World scene: a small set of obstacles (Sphere + Capsule + Box)
N_WORLD_SPHERES    = 4
N_WORLD_CAPSULES   = 3
N_WORLD_BOXES      = 2
N_WORLD_HALFSPACES = 2

# Neural SDF training defaults
NEURAL_SAMPLES    = 4000
NEURAL_EPOCHS     = 30
NEURAL_BATCH_SIZE = 512
NEURAL_LAYERS     = [128, 128, 128]

# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _gpu_monitor(interval_s: float = 0.02):
    """Background thread that samples GPU util & VRAM.  Yields a dict."""
    samples: dict[str, list[float]] = {"gpu_util": [], "vram_mb": []}
    stop_evt = threading.Event()

    def _sample() -> None:
        while not stop_evt.is_set():
            if _NVML_OK and _NVML_HANDLE is not None:
                util = _pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem  = _pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                samples["gpu_util"].append(float(util.gpu))
                samples["vram_mb"].append(float(mem.used) / 1024**2)
            stop_evt.wait(interval_s)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield samples
    finally:
        stop_evt.set()
        t.join(timeout=1.0)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label:      str
    batch_size: int
    op:         str           # "world" or "self"
    ms_call:    float         # wall-clock ms for the whole batched call
    ms_cfg:     float         # effective ms per config
    peak_gpu:   float = float("nan")
    peak_vram:  float = float("nan")

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, n_warmup=N_WARMUP, n_timed=N_TIMED) -> tuple:
    """Run fn(*args) n_warmup+n_timed times, return (last_result, median_s)."""
    out = None
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_timed):
        t0  = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))

def _time_fn_gpu(fn, *args, n_warmup=0, n_timed=N_TIMED) -> tuple:
    """Like _time_fn but also samples GPU util/VRAM.

    n_warmup defaults to 0 because callers are expected to drive their own
    warmup phase (with logging) before calling this function.
    """
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))

    out = None
    times = []
    with _gpu_monitor() as gpu_samples:
        for _ in range(n_timed):
            t0  = time.perf_counter()
            out = fn(*args)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

    return (
        out,
        float(np.median(times)),
        max(gpu_samples["gpu_util"], default=float("nan")),
        max(gpu_samples["vram_mb"],  default=float("nan")),
    )


def _warmup(fn, n: int = N_WARMUP) -> None:
    """Block until n calls of fn() complete (discards output)."""
    for _ in range(n):
        jax.block_until_ready(fn())

# ---------------------------------------------------------------------------
# World scene builder
# ---------------------------------------------------------------------------

def _make_world_scene(n_spheres: int, n_capsules: int, n_boxes: int,
                      n_halfspaces: int, rng):
    """Build a deterministic world scene with mixed obstacle types.

    Returns four CollGeom objects (each with M primitives batched on axis 0).
    """
    # Spheres scattered around the workspace
    centers_s = rng.uniform(-0.5, 0.5, (n_spheres, 3)).astype(np.float32)
    radii_s   = rng.uniform(0.05, 0.15, (n_spheres,)).astype(np.float32)
    world_spheres = Sphere.from_center_and_radius(
        center=jnp.array(centers_s),
        radius=jnp.array(radii_s),
    )

    # Capsules
    cap_centers  = rng.uniform(-0.4, 0.4, (n_capsules, 3)).astype(np.float32)
    cap_axes_raw = rng.standard_normal((n_capsules, 3)).astype(np.float32)
    cap_axes_raw /= np.linalg.norm(cap_axes_raw, axis=-1, keepdims=True)
    cap_heights  = rng.uniform(0.1, 0.3, n_capsules).astype(np.float32)
    cap_radii    = rng.uniform(0.03, 0.08, n_capsules).astype(np.float32)
    from pyronot.collision._geometry import Capsule as _Capsule
    world_capsules = _Capsule.from_radius_height(
        radius=jnp.array(cap_radii),
        height=jnp.array(cap_heights),
        position=jnp.array(cap_centers),
        wxyz=None,
    )

    # Boxes
    box_centers = rng.uniform(-0.3, 0.3, (n_boxes, 3)).astype(np.float32)
    box_lengths = rng.uniform(0.1, 0.3, (n_boxes, 3)).astype(np.float32)
    world_boxes = Box.from_center_and_dimensions(
        center=jnp.array(box_centers),
        length=jnp.array(box_lengths[:, 0]),
        width=jnp.array(box_lengths[:, 1]),
        height=jnp.array(box_lengths[:, 2]),
    )

    # HalfSpaces (planes)
    hs_points  = rng.uniform(-0.5, 0.5, (n_halfspaces, 3)).astype(np.float32)
    hs_normals = rng.standard_normal((n_halfspaces, 3)).astype(np.float32)
    hs_normals /= np.linalg.norm(hs_normals, axis=-1, keepdims=True)
    world_halfspaces = HalfSpace.from_point_and_normal(
        point=jnp.array(hs_points),
        normal=jnp.array(hs_normals),
    )

    return world_spheres, world_capsules, world_boxes, world_halfspaces

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _bench_backend(
    label: str,
    model,
    robot,
    world_geom,
    cfgs_by_batch: dict[int, jnp.ndarray],
    skip_world: bool = False,
    skip_self: bool = False,
    use_vmap: bool = False,
    world_op_tag: str = "world",
) -> list[BenchResult]:
    """Benchmark a collision model.

    use_vmap=True: wrap world/self calls with jax.jit(jax.vmap(...)).
    Required for JAX-native models that don't handle batched cfg internally
    (they use jdc.copy_and_mutate which enforces unbatched geometry shapes).
    CUDA models handle batching internally, so use_vmap=False.
    world_op_tag: op string for world results (e.g. "world-Sphere").
    """
    results = []

    for B, cfgs in cfgs_by_batch.items():
        # ── world collision ────────────────────────────────────────────────
        if not skip_world:
            try:
                if use_vmap:
                    fn_world_vmapped = jax.jit(jax.vmap(
                        lambda c: model.compute_world_collision_distance(robot, c, world_geom)
                    ))
                    fn_world = lambda c=cfgs: fn_world_vmapped(c)
                else:
                    fn_world = lambda c=cfgs: model.compute_world_collision_distance(
                        robot, c, world_geom
                    )
                logger.debug(f"    {label} world B={B}: warming up ({N_WARMUP}×)...")
                _warmup(fn_world)
                _, t_s, pk_gpu, pk_vram = _time_fn_gpu(fn_world)
                results.append(BenchResult(
                    label=label, batch_size=B, op=world_op_tag,
                    ms_call=t_s * 1e3, ms_cfg=t_s * 1e3 / B,
                    peak_gpu=pk_gpu, peak_vram=pk_vram,
                ))
            except Exception as exc:
                logger.warning(f"  {label} world B={B}: SKIPPED ({exc})")

        # ── self collision ─────────────────────────────────────────────────
        if not skip_self:
            try:
                if use_vmap:
                    fn_self_vmapped = jax.jit(jax.vmap(
                        lambda c: model.compute_self_collision_distance(robot, c)
                    ))
                    fn_self = lambda c=cfgs: fn_self_vmapped(c)
                else:
                    fn_self = lambda c=cfgs: model.compute_self_collision_distance(
                        robot, c
                    )
                logger.debug(f"    {label} self  B={B}: warming up ({N_WARMUP}×)...")
                _warmup(fn_self)
                _, t_s, pk_gpu, pk_vram = _time_fn_gpu(fn_self)
                results.append(BenchResult(
                    label=label, batch_size=B, op="self",
                    ms_call=t_s * 1e3, ms_cfg=t_s * 1e3 / B,
                    peak_gpu=pk_gpu, peak_vram=pk_vram,
                ))
            except Exception as exc:
                logger.warning(f"  {label} self B={B}: SKIPPED ({exc})")

    return results

# ---------------------------------------------------------------------------
# Table formatting  (mirrors bench_ik.py style)
# ---------------------------------------------------------------------------

_COL_W  = 22   # label column
_NUM_W  = 10   # numeric columns

def _hdr(cols: list[str]) -> str:
    row = f"  {'Method':<{_COL_W}}"
    for c in cols:
        row += f"  {c:>{_NUM_W}}"
    return row

def _sep(n: int) -> str:
    return "  " + "-" * (_COL_W + n * (_NUM_W + 2))

def _row(label: str, vals: list[str]) -> str:
    r = f"  {label:<{_COL_W}}"
    for v in vals:
        r += f"  {v:>{_NUM_W}}"
    return r

def _fmt_pct(v: float) -> str:
    return f"{v:.0f}%" if not np.isnan(v) else "n/a"

def _fmt_mb(v: float) -> str:
    return f"{v:.0f}" if not np.isnan(v) else "n/a"

def _print_table(title: str, results: list[BenchResult], op: str) -> None:
    rows = [r for r in results if r.op == op]
    if not rows:
        return

    batch_sizes = sorted({r.batch_size for r in rows})
    labels      = list(dict.fromkeys(r.label for r in rows))   # preserve order

    print(f"\n{'='*80}")
    print(f"  {title}  —  op={op}")
    print(f"{'='*80}")

    # One sub-table per batch size
    for B in batch_sizes:
        print(f"\n  Batch size = {B}")
        cols = ["ms/call", "ms/cfg", "gpu_pk(%)", "vram_pk(MB)"]
        print(_hdr(cols))
        print(_sep(len(cols)))

        for lbl in labels:
            row_data = [r for r in rows if r.label == lbl and r.batch_size == B]
            if not row_data:
                print(_row(lbl, ["—"] * len(cols)))
                continue
            r = row_data[0]
            vals = [
                f"{r.ms_call:.3f}",
                f"{r.ms_cfg:.4f}",
                _fmt_pct(r.peak_gpu),
                _fmt_mb(r.peak_vram),
            ]
            print(_row(lbl, vals))

def _print_speedup_table(
    results: list[BenchResult], op: str, baseline_label: str
) -> None:
    rows     = [r for r in results if r.op == op]
    labels   = list(dict.fromkeys(r.label for r in rows))
    batches  = sorted({r.batch_size for r in rows})

    # Index by (label, batch)
    by_key: dict[tuple, BenchResult] = {(r.label, r.batch_size): r for r in rows}

    print(f"\n  Speed-up vs {baseline_label!r}  (op={op})")
    col_hdrs = [f"B={B}" for B in batches]
    print(_hdr(col_hdrs))
    print(_sep(len(col_hdrs)))

    baseline = {B: by_key.get((baseline_label, B)) for B in batches}

    for lbl in labels:
        vals = []
        for B in batches:
            this = by_key.get((lbl, B))
            base = baseline[B]
            if this is None or base is None or this.ms_call <= 0:
                vals.append("—")
            else:
                vals.append(f"{base.ms_call / this.ms_call:.1f}×")
        print(_row(lbl, vals))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    rng = np.random.default_rng(42)

    print("=" * 80)
    print(f"Collision benchmark  (robot={ROBOT_NAME}, "
          f"n_warmup={N_WARMUP}, n_timed={N_TIMED})")
    gpu_status = "enabled" if _NVML_OK else "disabled (pip install nvidia-ml-py)"
    print(f"GPU monitoring: {gpu_status}")
    print("=" * 80)

    # ── Robot (capsule model) ──────────────────────────────────────────────
    print("\nLoading robot ...")
    urdf      = load_robot_description(f"{args.robot}_description")
    robot_cap = pk.Robot.from_urdf(urdf)
    n_act_cap = robot_cap.joints.num_actuated_joints
    lo_cap    = np.asarray(robot_cap.joints.lower_limits)
    hi_cap    = np.asarray(robot_cap.joints.upper_limits)
    print(f"  {args.robot}_description : {n_act_cap} actuated DOF")

    # ── Robot (sphere model) — separate URDF with sphere primitives ────────
    sph_urdf_path = args.spherized_urdf
    if sph_urdf_path.exists():
        urdf_sph  = yourdfpy.URDF.load(str(sph_urdf_path))
        robot_sph = pk.Robot.from_urdf(urdf_sph)
        n_act_sph = robot_sph.joints.num_actuated_joints
        lo_sph    = np.asarray(robot_sph.joints.lower_limits)
        hi_sph    = np.asarray(robot_sph.joints.upper_limits)
        print(f"  {sph_urdf_path.name}       : {n_act_sph} actuated DOF")
    else:
        print(f"  WARNING: {sph_urdf_path} not found; falling back to {args.robot}_description for spherized model")
        urdf_sph = urdf
        robot_sph = robot_cap
        lo_sph, hi_sph = lo_cap, hi_cap

    # ── Robot (coarse sphere model) — 1 sphere per link ───────────────────
    coarse_urdf_path = args.coarse_urdf
    if coarse_urdf_path.exists():
        urdf_coarse = yourdfpy.URDF.load(str(coarse_urdf_path))
        print(f"  {coarse_urdf_path.name} : coarse spherized URDF loaded")
    else:
        print(f"  WARNING: {coarse_urdf_path} not found; CUDA-Sphere-Coarse will be skipped")
        urdf_coarse = None

    # ── Collision models ───────────────────────────────────────────────────
    print("\nBuilding collision models ...")
    coll_cap = RobotCollision.from_urdf(urdf)
    print(f"  RobotCollision          : {coll_cap.num_links} links")

    coll_sph = RobotCollisionSpherized.from_urdf(urdf_sph)
    print(f"  RobotCollisionSpherized : {coll_sph.num_links} links")

    coll_sph_coarse = None
    if urdf_coarse is not None:
        coll_sph_coarse = RobotCollisionSpherized.from_urdf(urdf_coarse)
        print(f"  RobotCollisionSpherized (coarse) : {coll_sph_coarse.num_links} links")

    cuda_available = False
    cuda_cap = cuda_sph = cuda_sph_coarse = None
    try:
        cuda_cap = CUDARobotCollisionChecker(coll_cap)
        cuda_sph = CUDARobotCollisionChecker(coll_sph)
        if coll_sph_coarse is not None:
            cuda_sph_coarse = CUDARobotCollisionChecker(coll_sph, coarse_inner=coll_sph_coarse)
        cuda_available = True
        print("  CUDARobotCollisionChecker: OK (JAX FFI library loaded)")
        if cuda_sph_coarse is not None:
            print("  CUDARobotCollisionChecker (coarse-first): OK")
        else:
            print("  CUDARobotCollisionChecker (coarse-first): SKIP (no coarse URDF)")
    except RuntimeError as e:
        print(f"  CUDARobotCollisionChecker: SKIP ({e})")

    # ── World scene ────────────────────────────────────────────────────────
    print("\nBuilding world scene ...")
    world_spheres, world_capsules, world_boxes, world_halfspaces = _make_world_scene(
        N_WORLD_SPHERES, N_WORLD_CAPSULES, N_WORLD_BOXES, N_WORLD_HALFSPACES, rng
    )
    # Dict of primitive type → (geom, count) for per-primitive benchmarking
    world_primitives: dict[str, tuple] = {
        "Sphere":    (world_spheres,    N_WORLD_SPHERES),
        "Capsule":   (world_capsules,   N_WORLD_CAPSULES),
        "Box":       (world_boxes,      N_WORLD_BOXES),
        "HalfSpace": (world_halfspaces, N_WORLD_HALFSPACES),
    }
    # Use spheres as the default world_geom for neural training & self-collision
    world_geom = world_spheres
    print(f"  World obstacles: {N_WORLD_SPHERES} spheres, {N_WORLD_CAPSULES} capsules, "
          f"{N_WORLD_BOXES} boxes, {N_WORLD_HALFSPACES} halfspaces")

    # ── Random configs per batch size (one set per DOF count) ─────────────
    print("\nGenerating configs ...")
    max_B = max(BATCH_SIZES)
    cfgs_cap_np = rng.uniform(lo_cap, hi_cap, (max_B, lo_cap.shape[0])).astype(np.float32)
    cfgs_sph_np = rng.uniform(lo_sph, hi_sph, (max_B, lo_sph.shape[0])).astype(np.float32)

    cfgs_cap_by_batch: dict[int, jnp.ndarray] = {B: jnp.array(cfgs_cap_np[:B]) for B in BATCH_SIZES}
    cfgs_sph_by_batch: dict[int, jnp.ndarray] = {B: jnp.array(cfgs_sph_np[:B]) for B in BATCH_SIZES}

    # ── Neural SDF (one model per primitive type) ───────────────────────────
    neural_models: dict[str, NeuralRobotCollision] = {}
    if not args.skip_neural:
        print(f"\nTraining NeuralRobotCollision per primitive type "
              f"(samples={args.neural_samples}, epochs={NEURAL_EPOCHS}) ...")
        neural_base = NeuralRobotCollision.from_existing(
            coll_cap, key=jax.random.PRNGKey(0)
        )
        for i, (prim_name, (prim_geom, _)) in enumerate(world_primitives.items()):
            print(f"  Training on {prim_name} obstacles ...")
            try:
                neural_models[prim_name] = neural_base.train(
                    robot=robot_cap,
                    world_geom=prim_geom,
                    num_samples=args.neural_samples,
                    batch_size=NEURAL_BATCH_SIZE,
                    epochs=NEURAL_EPOCHS,
                    learning_rate=1e-3,
                    key=jax.random.PRNGKey(1 + i),
                    layer_sizes=NEURAL_LAYERS,
                )
            except Exception as exc:
                print(f"    Neural SDF training FAILED for {prim_name}: {exc}")
        if neural_models:
            print(f"  Neural SDF training complete ({len(neural_models)}/{len(world_primitives)} primitives).")

    # ── Run benchmarks ─────────────────────────────────────────────────────
    print("\nRunning benchmarks ...")

    all_results: list[BenchResult] = []

    # --- Self-collision benchmarks (primitive-independent) -----------------
    print("\n  Self-collision benchmarks ...")

    print("    JAX-Capsule self ...")
    all_results += _bench_backend(
        "JAX-Capsule", coll_cap, robot_cap, world_geom, cfgs_cap_by_batch,
        skip_world=True, use_vmap=True,
    )

    print("    JAX-Sphere self ...")
    all_results += _bench_backend(
        "JAX-Sphere", coll_sph, robot_sph, world_geom, cfgs_sph_by_batch,
        skip_world=True, use_vmap=True,
    )

    if neural_models:
        # Neural self-collision delegates to JAX-Capsule (same kernel)
        import copy
        for r in all_results:
            if r.label == "JAX-Capsule" and r.op == "self":
                nr = copy.copy(r)
                nr.label = "JAX-Neural"
                all_results.append(nr)

    if cuda_available:
        print("    CUDA-Capsule self ...")
        all_results += _bench_backend(
            "CUDA-Capsule", cuda_cap, robot_cap, world_geom, cfgs_cap_by_batch,
            skip_world=True,
        )
        print("    CUDA-Sphere self ...")
        all_results += _bench_backend(
            "CUDA-Sphere", cuda_sph, robot_sph, world_geom, cfgs_sph_by_batch,
            skip_world=True,
        )
        if cuda_sph_coarse is not None:
            print("    CUDA-Sphere-Coarse self ...")
            all_results += _bench_backend(
                "CUDA-Sphere-Coarse", cuda_sph_coarse, robot_sph, world_geom,
                cfgs_sph_by_batch, skip_world=True,
            )

    # --- World-collision benchmarks (per primitive type) --------------------
    for prim_name, (prim_geom, prim_count) in world_primitives.items():
        print(f"\n  World-collision benchmarks  (obstacle={prim_name}, M={prim_count}) ...")
        op_tag = f"world-{prim_name}"

        print(f"    JAX-Capsule ...")
        all_results += _bench_backend(
            "JAX-Capsule", coll_cap, robot_cap, prim_geom, cfgs_cap_by_batch,
            skip_self=True, use_vmap=True, world_op_tag=op_tag,
        )

        print(f"    JAX-Sphere ...")
        all_results += _bench_backend(
            "JAX-Sphere", coll_sph, robot_sph, prim_geom, cfgs_sph_by_batch,
            skip_self=True, use_vmap=True, world_op_tag=op_tag,
        )

        if prim_name in neural_models:
            print(f"    JAX-Neural ...")
            all_results += _bench_backend(
                "JAX-Neural", neural_models[prim_name], robot_cap, prim_geom,
                cfgs_cap_by_batch,
                skip_self=True, use_vmap=True, world_op_tag=op_tag,
            )

        if cuda_available:
            print(f"    CUDA-Capsule ...")
            all_results += _bench_backend(
                "CUDA-Capsule", cuda_cap, robot_cap, prim_geom, cfgs_cap_by_batch,
                skip_self=True, world_op_tag=op_tag,
            )
            print(f"    CUDA-Sphere ...")
            all_results += _bench_backend(
                "CUDA-Sphere", cuda_sph, robot_sph, prim_geom, cfgs_sph_by_batch,
                skip_self=True, world_op_tag=op_tag,
            )
            if cuda_sph_coarse is not None:
                print(f"    CUDA-Sphere-Coarse ...")
                all_results += _bench_backend(
                    "CUDA-Sphere-Coarse", cuda_sph_coarse, robot_sph, prim_geom,
                    cfgs_sph_by_batch, skip_self=True, world_op_tag=op_tag,
                )

    # ── Print tables ───────────────────────────────────────────────────────
    print("\n\n")
    for prim_name in world_primitives:
        op_tag = f"world-{prim_name}"
        _print_table(f"World collision distance  (obstacle={prim_name})", all_results, op_tag)
    _print_table("Self-collision distance",  all_results, "self")

    # Speed-up tables
    print("\n\n")
    print("=" * 80)
    print("  Speed-up summary")
    print("=" * 80)
    for prim_name in world_primitives:
        op_tag = f"world-{prim_name}"
        _print_speedup_table(all_results, op_tag, baseline_label="JAX-Capsule")
    _print_speedup_table(all_results, "self",  baseline_label="JAX-Capsule")

    # ── Numerical agreement check ──────────────────────────────────────────
    if cuda_available:
        print("\n\n" + "=" * 80)
        print("  Numerical agreement check  (batch=64, max abs diff)")
        print("=" * 80)
        cfg64_cap = cfgs_cap_by_batch[64]
        cfg64_sph = cfgs_sph_by_batch[64]

        checks = []

        # World collision checks — one per primitive type
        for prim_name, (prim_geom, _) in world_primitives.items():
            ref_cap = np.asarray(jax.vmap(
                lambda c, wg=prim_geom: coll_cap.compute_world_collision_distance(
                    robot_cap, c, wg)
            )(cfg64_cap))
            ref_sph = np.asarray(jax.vmap(
                lambda c, wg=prim_geom: coll_sph.compute_world_collision_distance(
                    robot_sph, c, wg)
            )(cfg64_sph))

            checks.append((
                f"CUDA-Cap world-{prim_name}",
                np.asarray(cuda_cap.compute_world_collision_distance(
                    robot_cap, cfg64_cap, prim_geom)),
                ref_cap, False,
            ))
            checks.append((
                f"CUDA-Sph world-{prim_name}",
                np.asarray(cuda_sph.compute_world_collision_distance(
                    robot_sph, cfg64_sph, prim_geom)),
                ref_sph, False,
            ))

        # Self collision checks
        ref_self_cap = np.asarray(jax.vmap(
            lambda c: coll_cap.compute_self_collision_distance(robot_cap, c)
        )(cfg64_cap))
        ref_self_sph = np.asarray(jax.vmap(
            lambda c: coll_sph.compute_self_collision_distance(robot_sph, c)
        )(cfg64_sph))

        checks += [
            ("CUDA-Capsule self",  np.asarray(
                cuda_cap.compute_self_collision_distance(robot_cap, cfg64_cap)),
             ref_self_cap, False),
            # Note: CUDA-Sphere self computes the true S×S minimum over all sphere pairs
            # between each link pair, while the JAX reference only computes the diagonal
            # (sphere s of link i vs sphere s of link j).  CUDA is more accurate; the
            # discrepancy is expected and not a bug.
            ("CUDA-Sphere self",   np.asarray(
                cuda_sph.compute_self_collision_distance(robot_sph, cfg64_sph)),
             ref_self_sph, True),
        ]

        for name, arr, ref, cross_reduction in checks:
            if arr.shape == ref.shape:
                valid = (arr < 1e8) & (ref < 1e8)
                if valid.any():
                    diff = float(np.abs(arr[valid] - ref[valid]).max())
                    n_masked = int((~valid).sum())
                    note = f"  ({n_masked} sentinel pairs masked)" if n_masked else ""
                    if cross_reduction:
                        note += "  [CUDA uses full SxS min; JAX uses diagonal — expected diff]"
                    print(f"  {name:<30}  max|Δ| = {diff:.4e}{note}")
                else:
                    print(f"  {name:<30}  all pairs are padding sentinels, skipped")
            else:
                print(f"  {name:<30}  shape {arr.shape} vs ref {ref.shape} (models differ)")

        # ── Coarse-first coverage report ───────────────────────────────────
        if cuda_sph_coarse is not None:
            print("\n\n" + "=" * 80)
            print("  CUDA-Sphere-Coarse  —  coarse guard coverage  (batch=64)")
            print("  Configs where coarse says clear → fine kernel skipped (+1 returned).")
            print("  Configs where coarse says collision → fine kernel runs (exact distances).")
            print("=" * 80)
            for prim_name, (prim_geom, _) in world_primitives.items():
                coarse_result = np.asarray(
                    cuda_sph_coarse.compute_world_collision_distance(
                        robot_sph, cfg64_sph, prim_geom
                    )
                )  # [64, N, M]
                # A batch element was flagged clear by coarse if ALL distances == 1.0
                clear_mask = np.all(coarse_result == 1.0, axis=(-1, -2))
                n_clear = int(clear_mask.sum())
                n_total = clear_mask.shape[0]
                print(f"  world-{prim_name:<10}  coarse-clear: {n_clear}/{n_total} "
                      f"({100*n_clear/n_total:.0f}%)"
                      f"  — fine kernel skipped for {n_clear} configs")

            # Self-collision coarse coverage
            coarse_self = np.asarray(
                cuda_sph_coarse.compute_self_collision_distance(robot_sph, cfg64_sph)
            )  # [64, P]
            clear_self = np.all(coarse_self == 1.0, axis=-1)
            n_clear_self = int(clear_self.sum())
            n_total_self = clear_self.shape[0]
            print(f"  self           coarse-clear: {n_clear_self}/{n_total_self} "
                  f"({100*n_clear_self/n_total_self:.0f}%)"
                  f"  — fine kernel skipped for {n_clear_self} configs")

    print("\nDone.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot",           default=ROBOT_NAME,
                        help="Robot name for robot_descriptions (default: panda)")
    parser.add_argument("--spherized-urdf",  default=str(SPHERIZED_URDF),
                        type=pathlib.Path,
                        help="Path to the spherized URDF for RobotCollisionSpherized "
                             f"(default: {SPHERIZED_URDF})")
    parser.add_argument("--coarse-urdf",     default=str(COARSE_SPHERIZED_URDF),
                        type=pathlib.Path,
                        help="Path to the coarse spherized URDF for CUDA-Sphere-Coarse "
                             f"(default: {COARSE_SPHERIZED_URDF})")
    parser.add_argument("--skip-neural",     action="store_true",
                        help="Skip neural SDF training and benchmarking")
    parser.add_argument("--neural-samples",  type=int, default=NEURAL_SAMPLES,
                        help=f"Training set size for neural SDF (default: {NEURAL_SAMPLES})")
    args = parser.parse_args()
    main(args)
