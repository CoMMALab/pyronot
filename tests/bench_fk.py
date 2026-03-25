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
           bash src/pyronot/cuda_kernels/build_fk_cuda.sh
    3. Local spherized URDFs must be present under ``resources/``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyronot as pk
import yourdfpy

# ---------------------------------------------------------------------------
# Configuration (matched to tests/test_fk_cuda.py)
# ---------------------------------------------------------------------------
ROBOT_NAMES = ("panda", "fetch", "baxter")
RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"
ROBOT_URDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda_spherized.urdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch_spherized.urdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter_spherized.urdf",
}
BATCH_SIZES = [1, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
N_WARMUP = 5
N_TIMED = 50
N_DEVICE_REPEATS = 8
ATOL = 1e-4
RTOL = 1e-4


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


def _time_device_scan(timer_fn, cfg_seq: jax.Array, n: int = N_TIMED) -> float:
    """Return median per-FK wall-clock time (seconds), amortized on-device."""
    repeats = int(cfg_seq.shape[0])
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = timer_fn(cfg_seq)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / repeats)
    return float(np.median(times))


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

    rows: list[dict[str, float | int | bool]] = []
    all_passed = True

    print("\n" + "=" * 70)
    print(f"FK correctness & performance: JAX vs CUDA  ({robot_name} robot)")
    print("=" * 70)
    print(
        "  {0:<8} {1:<8} {2:>12}   {3:>12}   {4:>8}   {5}".format(
            "Impl", "Batch", "JAX (ms)", "CUDA (ms)", "Speedup", "Max |err|"
        )
    )
    print("-" * 70)

    for batch in BATCH_SIZES:
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

        t_jax = _time_device_scan(timer_jax, cfg_seq_jax)
        t_cuda = _time_device_scan(timer_cuda, cfg_seq_jax)
        speedup = t_jax / t_cuda if t_cuda > 0 else float("nan")

        status = "OK" if passed else "FAIL"
        print(
            f"  {'JAX':<8} {batch:<8} {t_jax * 1e3:>12.3f}   "
            f"{t_cuda * 1e3:>12.3f}   {speedup:>8.2f}x   "
            f"|err|={max_err:.2e}  [{status}]"
        )

        rows.append(
            {
                "batch": batch,
                "jax_ms": t_jax * 1e3,
                "cuda_ms": t_cuda * 1e3,
                "speedup": speedup,
                "max_abs_err": max_err,
                "passed": passed,
            }
        )

    print("-" * 70)
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
            fieldnames=["batch", "jax_ms", "cuda_ms", "speedup", "max_abs_err", "passed"],
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
