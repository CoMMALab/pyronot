"""Benchmark and correctness test: CUDA FK vs JAX FK (CSV export).

Repeats the experiments from ``tests/test_fk_cuda.py`` for three robots:
``panda``, ``fetch``, and ``baxter``.

For each robot, writes one CSV file with per-batch metrics:
    - batch
    - jax_ms
    - cuda_ms
    - speedup
    - max_abs_err
    - passed

Usage:
    python tests/bench_fk.py
    python tests/bench_fk.py --outdir results

Prerequisites:
    1. A CUDA-capable GPU must be available.
    2. The CUDA FK library must be compiled:
           bash src/pyroffi/cuda_kernels/build_fk_cuda.sh
    3. Local spherized URDFs must be present under ``resources/``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import pynvml
import pyroffi as pk
import yourdfpy

# ---------------------------------------------------------------------------
# Configuration (matched to tests/test_fk_cuda.py)
# ---------------------------------------------------------------------------
ROBOT_NAMES = ("panda", "fetch", "baxter", "g1")
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
ROBOT_URDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda_spherized.urdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch_spherized.urdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter_spherized.urdf",
    "g1": RESOURCE_ROOT / "g1_description" / "g1_29dof_with_hand_rev_1_0_spherized.urdf",
}

# Per-robot batch sizes: larger robots use more shared memory per batch item,
# so their max feasible batch is smaller.  Smaller robots can go higher.
ROBOT_BATCH_SIZES = {
    "panda":  [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    "fetch":  [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    "baxter": [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
    "g1":     [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
}
DEFAULT_BATCH_SIZES = [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

N_WARMUP = 5
N_TIMED = 50
N_DEVICE_REPEATS = 5
ATOL = 1e-4
RTOL = 1e-4

# ---------------------------------------------------------------------------
# GPU utilization sampler
# ---------------------------------------------------------------------------
pynvml.nvmlInit()
_GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)


class GpuUtilSampler:
    """Background thread that samples GPU utilization at a fixed interval."""

    def __init__(self, interval: float = 0.01):
        self._interval = interval
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuUtilSampler":
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            self._samples.append(util.gpu)
            self._stop.wait(self._interval)

    def stop(self) -> float:
        """Stop sampling and return mean GPU utilization (%)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))


def _build_device_timer(
    fk_fn,
):
    """Build a jitted timer function that runs FK repeatedly on-device.

    The returned function executes FK for each entry in ``cfg_seq`` via
    ``lax.scan``, so one host dispatch covers many FK evaluations.
    """

    @jax.jit
    def _timer(cfg_seq: jax.Array) -> jax.Array:
        def _body(carry: jax.Array, cfg_i: jax.Array) -> tuple[jax.Array, None]:
            out = fk_fn(cfg_i)
            # Tiny dependency to keep FK calls live in the scan body.
            contrib = out[0, 0, 0].astype(carry.dtype)
            return carry + contrib, None

        checksum, _ = jax.lax.scan(_body, jnp.float32(0.0), cfg_seq)
        return checksum

    return _timer


def _time_device_scan(timer_fn, cfg_seq: jax.Array, n: int = N_TIMED) -> tuple[float, float]:
    """Return (median per-FK wall-clock time in seconds, mean GPU util %)."""
    repeats = int(cfg_seq.shape[0])
    times: list[float] = []
    sampler = GpuUtilSampler().start()
    for _ in range(n):
        t0 = time.perf_counter()
        out = timer_fn(cfg_seq)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / repeats)
    gpu_util = sampler.stop()
    return float(np.median(times)), gpu_util


def _compute_kernel_config(robot: "pk.Robot") -> tuple[int, int, int]:
    """Compute the warp-packing config the CUDA kernel will auto-select.

    Returns (max_level_width, items_per_warp, lanes_per_item).
    """
    starts = np.array(robot.joints.fk_level_starts)
    n_levels = robot.joints.num_fk_levels
    max_w = 0
    for lvl in range(n_levels):
        w = int(starts[lvl + 1] - starts[lvl])
        if w > max_w:
            max_w = w
    if max_w <= 2:
        ipw = 16
    elif max_w <= 4:
        ipw = 8
    elif max_w <= 8:
        ipw = 4
    elif max_w <= 16:
        ipw = 2
    else:
        ipw = 1
    return max_w, ipw, 32 // ipw


def _run_robot_benchmark(robot_name: str) -> tuple[list[dict[str, float | int | bool]], bool]:
    """Run FK correctness/timing benchmark for one robot."""
    urdf_path = ROBOT_URDFS[robot_name]
    if not urdf_path.exists():
        raise FileNotFoundError(f"Spherized URDF not found: {urdf_path}")
    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints

    fk_jax = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=False))
    fk_cuda = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=True))
    timer_jax = _build_device_timer(fk_jax)
    timer_cuda = _build_device_timer(fk_cuda)

    rng = np.random.default_rng(42)
    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)

    max_w, ipw, lpi = _compute_kernel_config(robot)
    batch_sizes = ROBOT_BATCH_SIZES.get(robot_name, DEFAULT_BATCH_SIZES)

    rows: list[dict[str, float | int | bool]] = []
    all_passed = True

    print("\n" + "=" * 78)
    print(f"FK correctness & performance: JAX vs CUDA  ({robot_name} robot)")
    print(f"  joints={robot.joints.num_joints}  actuated={n_act}"
          f"  levels={robot.joints.num_fk_levels}"
          f"  max_level_width={max_w}")
    print(f"  kernel config: ITEMS_PER_WARP={ipw}  LANES_PER_ITEM={lpi}")
    print("=" * 78)
    print(
        "  {0:<8} {1:<8} {2:>12}   {3:>12}   {4:>8}   {5:>10}   {6:>10}   {7}".format(
            "Impl", "Batch", "JAX (ms)", "CUDA (ms)", "Speedup",
            "JAX GPU%", "CUDA GPU%", "Max |err|"
        )
    )
    print("-" * 78)

    for batch in batch_sizes:
        cfg_np = rng.uniform(lo, hi, size=(batch, n_act)).astype(np.float32)
        cfg_jax = jax.device_put(jnp.array(cfg_np))
        cfg_seq_np = rng.uniform(
            lo, hi, size=(N_DEVICE_REPEATS, batch, n_act)
        ).astype(np.float32)
        cfg_seq_jax = jax.device_put(jnp.array(cfg_seq_np))

        # Ensure host->device transfers are complete before warmup/timing.
        jax.block_until_ready(cfg_jax)
        jax.block_until_ready(cfg_seq_jax)

        for _ in range(N_WARMUP):
            jax.block_until_ready(fk_jax(cfg_jax))
            jax.block_until_ready(fk_cuda(cfg_jax))
            jax.block_until_ready(timer_jax(cfg_seq_jax))
            jax.block_until_ready(timer_cuda(cfg_seq_jax))

        out_jax = np.array(fk_jax(cfg_jax))
        out_cuda = np.array(fk_cuda(cfg_jax))
        max_err = float(np.abs(out_jax - out_cuda).max())
        passed = bool(np.allclose(out_jax, out_cuda, atol=ATOL, rtol=RTOL))
        all_passed &= passed

        t_jax, gpu_jax = _time_device_scan(timer_jax, cfg_seq_jax)
        t_cuda, gpu_cuda = _time_device_scan(timer_cuda, cfg_seq_jax)
        speedup = t_jax / t_cuda if t_cuda > 0 else float("nan")

        status = "OK" if passed else "FAIL"
        print(
            f"  {'JAX':<8} {batch:<8} {t_jax * 1e3:>12.3f}   "
            f"{t_cuda * 1e3:>12.3f}   {speedup:>8.2f}x   "
            f"{gpu_jax:>9.1f}%   {gpu_cuda:>9.1f}%   "
            f"|err|={max_err:.2e}  [{status}]"
        )

        rows.append(
            {
                "batch": batch,
                "jax_ms": t_jax * 1e3,
                "cuda_ms": t_cuda * 1e3,
                "speedup": speedup,
                "jax_gpu_util": gpu_jax,
                "cuda_gpu_util": gpu_cuda,
                "max_abs_err": max_err,
                "passed": passed,
            }
        )

    print("-" * 78)
    if all_passed:
        print(f"PASSED: outputs agree within tolerance (atol={ATOL}, rtol={RTOL}).")
    else:
        print("FAILED: one or more batch sizes exceeded tolerance.")

    return rows, all_passed


def _write_csv(out_path: pathlib.Path, rows: list[dict[str, float | int | bool]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["batch", "jax_ms", "cuda_ms", "speedup", "jax_gpu_util", "cuda_gpu_util", "max_abs_err", "passed"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="FK benchmark with CSV export")
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory to write CSV files into (default: current directory).",
    )
    args = parser.parse_args()

    overall_ok = True
    for robot_name in ROBOT_NAMES:
        rows, ok = _run_robot_benchmark(robot_name)
        overall_ok &= ok

        out_file = args.outdir / f"bench_fk_{robot_name}.csv"
        _write_csv(out_file, rows)
        print(f"Wrote {out_file}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()