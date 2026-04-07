"""Region-based Inverse Kinematics using Stein Variational Gradient Descent (SVGD).

Region-based IK uses SVGD to sample a distribution over joint configurations
whose end-effectors cover a specified box region. The algorithm transports
particles to cover the constraint manifold uniformly, avoiding boundary
clustering that plagues random sampling methods.

Key algorithms:
  - SVGD: Particle-based density transport with RBF kernel repulsion
  - Jacobian-guided SVGD: Incorporates task-space gradients for convergence

References:
  - Liu, Qiang, and Dilin Wang. "Stein variational gradient descent."
    Advances in neural information processing systems 30 (2016).
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot

_REGION_IK_MAX_TPB_BY_SMEM = 384  # MAX_JOINTS=64, MAX_ACT=16


def _validate_threads_per_block(threads_per_block: int) -> None:
    if (
        threads_per_block < 32
        or threads_per_block > 1024
        or threads_per_block % 32 != 0
    ):
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
    if threads_per_block > _REGION_IK_MAX_TPB_BY_SMEM:
        raise ValueError(
            f"threads_per_block={threads_per_block} exceeds the current shared-memory "
            f"budget for svgd_region_ik_cuda; use <= {_REGION_IK_MAX_TPB_BY_SMEM}."
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "n_iters",
        "bandwidth",
        "step_size",
        "threads_per_block",
    ),
)
def _svgd_region_batch_select_jit(
    seeds: Array,
    init_points: Array,
    twists: Array,
    parent_tf: Array,
    parent_idx: Array,
    act_idx: Array,
    mimic_mul: Array,
    mimic_off: Array,
    mimic_act_idx: Array,
    topo_inv: Array,
    ancestor_mask: Array,
    box_min: Array,
    box_max: Array,
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    n_iters: int,
    bandwidth: float,
    step_size: float,
    threads_per_block: int = 128,
) -> tuple[Array, Array, Array, Array, Array]:
    """JIT-compiled CUDA batch solve + per-target restart winner selection."""
    from ..cuda_kernels._svgd_region_ik_cuda import svgd_region_ik_cuda

    cfgs, errs, ee_points, target_points = svgd_region_ik_cuda(
        seeds=seeds,
        init_points=init_points,
        twists=twists,
        parent_tf=parent_tf,
        parent_idx=parent_idx,
        act_idx=act_idx,
        mimic_mul=mimic_mul,
        mimic_off=mimic_off,
        mimic_act_idx=mimic_act_idx,
        topo_inv=topo_inv,
        target_jnts=jnp.array([target_jnt], dtype=jnp.int32),
        ancestor_masks=ancestor_mask[None, :],
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        n_iters=n_iters,
        bandwidth=bandwidth,
        step_size=step_size,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    inside = jnp.all(
        (best_ee >= box_min[None, :]) & (best_ee <= box_max[None, :]), axis=1
    )
    return best_cfgs, best_ee, best_targets, best_errs, inside


def _compute_ancestor_mask(robot: Robot, target_link_index: int) -> Tuple[int, Array]:
    parent_joint_indices = np.asarray(robot.links.parent_joint_indices, dtype=np.int32)
    parent_idx = np.asarray(robot.joints.parent_indices, dtype=np.int32)
    n_joints = int(robot.joints.num_joints)

    target_jnt = int(parent_joint_indices[target_link_index])
    if target_jnt < 0:
        raise ValueError(
            f"Target link index {target_link_index} maps to root/base (no parent joint)."
        )

    mask = np.zeros((n_joints,), dtype=np.int32)
    j = target_jnt
    while j >= 0:
        mask[j] = 1
        j = int(parent_idx[j])

    return target_jnt, jnp.array(mask, dtype=jnp.int32)


def _seeds_per_launch_budget(
    n_act: int,
    desired: int,
    memory_limit_gb: float,
    restarts_per_target: int,
) -> int:
    bytes_per_seed = max(4096, 4 * (3 * n_act + n_act + 3 + 3 + 1 + 32))
    bytes_per_target = bytes_per_seed * max(1, int(restarts_per_target))
    budget = int((memory_limit_gb * (1024**3)) // bytes_per_target)
    budget = max(1, budget)
    return max(1, min(desired, budget))


def _box_entropy(
    ee_points: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Shannon entropy (nats) of the EE-point distribution.

    Points are normalized to [0, 1]^3 within the box and binned into an
    n_bins^3 histogram.  Maximum entropy for a uniform distribution is
    log(n_bins^3) nats (≈ 6.91 for n_bins=10).
    """
    span = box_max - box_min
    normalized = (ee_points - box_min) / np.maximum(span, 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    hist, _ = np.histogramdd(normalized, bins=n_bins, range=[[0.0, 1.0]] * 3)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log(p)))


def svgd_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"],
    box_max: Float[Array, "3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    n_iters: int = 50,
    bandwidth: float = 0.1,
    step_size: float = 0.05,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
) -> Tuple[
    Float[Array, "n_samples n_act"],
    Float[Array, "n_samples 3"],
    Float[Array, "n_samples 3"],
    Float[Array, "n_samples"],
]:
    """Sample IK configurations whose end-effectors cover a box region using SVGD.

    The CUDA kernel uses SVGD to transport particles and cover the constraint
    manifold uniformly. Each seed starts from a configuration and is updated
    using:
      - Gradient of log-target (task-space residuals)
      - RBF kernel repulsion for manifold coverage

    Args:
        robot: Robot model.
        target_link_index: End-effector link index.
        rng_key: JAX PRNG key.
        previous_cfg: Previous joint configuration (warm-start).
        box_min: Box minimum corner [x, y, z].
        box_max: Box maximum corner [x, y, z].
        num_samples: Total number of samples to collect.
        seeds_per_launch: Seeds per CUDA kernel launch.
        restarts_per_target: Warm restarts per target point.
        n_iters: Number of SVGD iterations.
        bandwidth: RBF kernel bandwidth.
        step_size: SVGD step size.
        threads_per_block: CUDA threads per block (multiple of 32).
        fixed_joint_mask: Mask for fixed joints.
        memory_limit_gb: Memory budget in GB.
        max_batches: Maximum batches to run.
        target_entropy: If provided, stop when entropy reaches this value.
        entropy_bins: Number of histogram bins per axis.
        verbose: Print timing info.

    Returns:
        Tuple of (cfgs, ee_points, target_points, errors) where cfgs has shape
        (n_samples, n_act), ee_points and target_points have shape (n_samples, 3),
        and errors has shape (n_samples,).
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_mask = jnp.zeros((n_act,), dtype=jnp.int32)
    else:
        fixed_mask = fixed_joint_mask.astype(jnp.int32)

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)

    box_min_arr = jnp.asarray(box_min, dtype=jnp.float32)
    box_max_arr = jnp.asarray(box_max, dtype=jnp.float32)
    if not bool(jnp.all(box_max_arr > box_min_arr)):
        raise ValueError("box_max must be strictly greater than box_min for all axes.")

    box_min_np = np.asarray(box_min_arr)
    box_max_np = np.asarray(box_max_arr)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block))
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    cfg_chunks: list[Array] = []
    ee_chunks: list[Array] = []
    tgt_chunks: list[Array] = []
    err_chunks: list[Array] = []

    collected = 0
    attempts = 0
    if max_batches is None:
        max_batches = max(
            8, 8 * int(np.ceil(num_samples / (seeds_per_launch * restarts_per_target)))
        )

    carry_cfg = previous_cfg
    key = rng_key

    while collected < num_samples and attempts < max_batches:
        attempts += 1
        n_batch = seeds_per_launch

        if verbose:
            import time as _time

            _t_batch = _time.perf_counter()

        key, key_warm, key_rand, key_pts, key_seed = jax.random.split(key, 5)

        n_total = n_batch * restarts_per_target
        n_warm = max(1, n_total // 4)
        n_rand = n_total - n_warm

        warm = jnp.clip(
            carry_cfg[None, :] + jax.random.normal(key_warm, (n_warm, n_act)) * 0.05,
            lower,
            upper,
        )
        rand = jax.random.uniform(key_rand, (n_rand, n_act), minval=lower, maxval=upper)
        seeds_flat = jnp.concatenate([warm, rand], axis=0)
        seeds_flat = jnp.where(fixed_mask[None, :], carry_cfg[None, :], seeds_flat)
        seeds = seeds_flat.reshape(n_batch, restarts_per_target, n_act)

        init_points_base = jax.random.uniform(
            key_pts,
            (n_batch, 3),
            minval=box_min,
            maxval=box_max,
            dtype=jnp.float32,
        )
        init_points = jnp.repeat(
            init_points_base[:, None, :], restarts_per_target, axis=1
        )

        rng_seed = jax.random.randint(
            key_seed,
            (),
            minval=1,
            maxval=np.iinfo(np.int32).max,
            dtype=jnp.int32,
        )

        try:
            best_cfgs, best_ee, best_targets, best_errs, inside = (
                _svgd_region_batch_select_jit(
                    seeds=seeds,
                    init_points=init_points,
                    twists=robot.joints.twists,
                    parent_tf=robot.joints.parent_transforms,
                    parent_idx=robot.joints.parent_indices,
                    act_idx=robot.joints.actuated_indices,
                    mimic_mul=robot.joints.mimic_multiplier,
                    mimic_off=robot.joints.mimic_offset,
                    mimic_act_idx=robot.joints.mimic_act_indices,
                    topo_inv=robot.joints._topo_sort_inv,
                    ancestor_mask=ancestor_mask,
                    box_min=box_min,
                    box_max=box_max,
                    lower=lower,
                    upper=upper,
                    fixed_mask=fixed_mask,
                    rng_seed=rng_seed,
                    target_jnt=target_jnt,
                    n_iters=n_iters,
                    bandwidth=bandwidth,
                    step_size=step_size,
                    threads_per_block=threads_per_block,
                )
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower() and seeds_per_launch > 1:
                seeds_per_launch = max(1, seeds_per_launch // 2)
                continue
            raise

        valid_count = int(jnp.sum(inside))

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            print(
                f"  batch {attempts:3d}: {valid_count:4d}/{n_batch} valid "
                f"({valid_count / n_batch * 100:.1f}%), "
                f"collected {collected + valid_count}/{num_samples}, "
                f"{_batch_ms:.1f} ms"
            )

        if valid_count > 0:
            cfg_chunks.append(best_cfgs[inside])
            ee_chunks.append(best_ee[inside])
            tgt_chunks.append(best_targets[inside])
            err_chunks.append(best_errs[inside])
            collected += valid_count
            carry_cfg = best_cfgs[inside][-1]
        else:
            carry_cfg = best_cfgs[int(jnp.argmin(best_errs))]

        if target_entropy is not None and len(ee_chunks) > 0:
            ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks], axis=0)
            current_entropy = _box_entropy(
                ee_so_far, box_min_np, box_max_np, entropy_bins
            )
            if current_entropy >= target_entropy:
                break

    if target_entropy is None and collected < num_samples:
        import warnings
        warnings.warn(
            f"Unable to collect enough in-box IK samples. "
            f"Collected {collected}/{num_samples} after {attempts} batches. "
            "Try increasing max_iter/restarts_per_target or widening the box.",
            stacklevel=2,
        )
        if not cfg_chunks:
            raise RuntimeError("No valid in-box samples collected at all.")

    cfg_all = jnp.concatenate(cfg_chunks, axis=0)[:num_samples]
    ee_all = jnp.concatenate(ee_chunks, axis=0)[:num_samples]
    tgt_all = jnp.concatenate(tgt_chunks, axis=0)[:num_samples]
    err_all = jnp.concatenate(err_chunks, axis=0)[:num_samples]

    return cfg_all, ee_all, tgt_all, err_all
