"""Region-constrained IK sampling via CUDA.

This module provides two of the three region-IK samplers:

  brownian_motion_sample_box_region_cuda  (this file)
    Two-phase per-seed strategy: Levenberg-Marquardt boundary reach, then
    null-space Brownian shuffling with periodic FK-check corrections.  Best
    for dense, well-distributed coverage of a single region.

  svgd_sample_box_region_cuda  (this file)
    Particle transport via Stein Variational Gradient Descent with RBF-kernel
    repulsion.  Tends to spread particles more uniformly than random sampling
    and avoids boundary clustering.

  hit_and_run_sample_box_region_cuda  (this file)
    Two-phase per-seed strategy: LM boundary reach, then Markov-chain
    hit-and-run Gaussian perturbations in joint space.  Lighter per-step cost
    than Brownian motion; good for exploration when the box is large.

Batched box queries
-------------------
All three samplers accept ``box_min`` / ``box_max`` either as

  * shape ``(3,)``      → single box; returns ``(num_samples, ...)``
  * shape ``(n_boxes, 3)`` → one box per row; returns ``(n_boxes, num_samples, ...)``

Seeds in each kernel launch are distributed round-robin across boxes so every
region is explored in parallel on the GPU.  Each box independently accumulates
``num_samples`` valid configurations before the call returns.

Example — two boxes in one call::

    cfgs, ee, tgt, err = brownian_motion_sample_box_region_cuda(
        robot, ee_link, rng_key, prev_cfg,
        box_min=jnp.array([[0.3, -0.2, 0.1], [0.3, 0.1, 0.3]]),
        box_max=jnp.array([[0.5,  0.0, 0.3], [0.5, 0.3, 0.5]]),
        num_samples=512,
    )
    # cfgs.shape == (2, 512, n_act)
    # ee.shape   == (2, 512, 3)

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

_REGION_IK_MAX_TPB_BY_SMEM = 384  # MAX_JOINTS=64, MAX_ACT=16 build of CUDA kernel.


def _validate_threads_per_block(threads_per_block: int, max_tpb: int = _REGION_IK_MAX_TPB_BY_SMEM) -> None:
    if threads_per_block < 32 or threads_per_block > 1024 or threads_per_block % 32 != 0:
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
    if threads_per_block > max_tpb:
        raise ValueError(
            f"threads_per_block={threads_per_block} exceeds the shared-memory "
            f"budget for this kernel; use <= {max_tpb}."
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "noise_std",
        "n_brownian_steps",
        "fk_check_freq",
        "threads_per_block",
    ),
)
def _brownian_motion_batch_select_jit(
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
    box_mins: Array,   # (n_problems, 3) — per-problem box mins
    box_maxs: Array,   # (n_problems, 3) — per-problem box maxs
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    max_iter: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
    eps_pos: float,
    noise_std: float,
    n_brownian_steps: int,
    fk_check_freq: int,
    threads_per_block: int = 128,
) -> tuple[Array, Array, Array, Array, Array]:
    """JIT-compiled CUDA batch solve + per-target restart winner selection."""
    from ..cuda_kernels._brownian_motion_ik_cuda import brownian_motion_ik_cuda

    cfgs, errs, ee_points, target_points = brownian_motion_ik_cuda(
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
        ancestor_mask=ancestor_mask,
        target_quat=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        box_mins=box_mins,
        box_maxs=box_maxs,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        rng_seed=rng_seed,
        target_jnt=target_jnt,
        max_iter=max_iter,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        noise_std=noise_std,
        n_brownian_steps=n_brownian_steps,
        fk_check_freq=fk_check_freq,
        threads_per_block=threads_per_block,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    # Per-problem inside check using per-problem box bounds.
    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, best_targets, best_errs, inside


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
    box_mins: Array,   # (n_problems, 3) — per-problem box mins
    box_maxs: Array,   # (n_problems, 3) — per-problem box maxs
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
    """JIT-compiled SVGD CUDA batch solve + per-target restart winner selection."""
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
    # Per-problem inside check using per-problem box bounds.
    inside = jnp.all(
        (best_ee >= box_mins) & (best_ee <= box_maxs), axis=1
    )
    return best_cfgs, best_ee, best_targets, best_errs, inside


def _compute_ancestor_mask(robot: Robot, target_link_index: int) -> tuple[int, Array]:
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
    # Conservative per-seed estimate for input/output buffers and temporary launch overhead.
    # Local register/shared memory is not part of VRAM accounting, but we keep a large safety margin.
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
    """Shannon entropy (nats) of the EE-point distribution discretized inside the box.

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


# ── Brownian-motion sampler ──────────────────────────────────────────────────


def brownian_motion_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    max_iter: int = 20,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    noise_std: float = 0.02,
    n_brownian_steps: int = 100,
    fk_check_freq: int = 5,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Sample IK configurations whose end-effectors lie inside one or more box regions.

    The CUDA kernel uses a two-phase strategy per seed:

      Phase 1 – LM Boundary Reach (``max_iter`` iterations):
        Levenberg-Marquardt IK toward a fixed target pre-sampled inside the
        box.  Exits early once the position residual is below ``eps_pos``.

      Phase 2 – Null-Space Brownian Shuffle (``n_brownian_steps`` steps):
        Gaussian perturbations in joint space (std = ``noise_std``) projected
        onto the null-space of the position Jacobian.  Every ``fk_check_freq``
        steps an FK check is run: if the EE has drifted outside the box, 2
        corrective LM steps push it back to the nearest box-boundary point.

    **Batched boxes**: pass ``box_min``/``box_max`` with shape ``(n_boxes, 3)``
    to sample ``num_samples`` configurations for *each* box in a single call.
    Seeds within each launch are distributed round-robin across boxes so all
    regions are explored in parallel.  Results have a leading ``n_boxes``
    dimension; a single-box call (shape ``(3,)``) preserves the original
    ``(num_samples, ...)`` return shape.

    Args:
        box_min: Box minimum corner(s).  Shape ``(3,)`` for a single box or
            ``(n_boxes, 3)`` for multiple boxes.
        box_max: Box maximum corner(s).  Matching shape to ``box_min``.
        max_iter: Maximum LM iterations in Phase 1 (boundary reach).
        noise_std: Standard deviation of Brownian joint perturbations in Phase 2.
        n_brownian_steps: Number of Brownian steps in Phase 2.
        fk_check_freq: FK check / corrective-LM period during Phase 2.
        target_entropy: If provided, sampling stops early once the Shannon entropy
            of the collected EE-point distribution reaches this value (nats).
        entropy_bins: Number of histogram bins per axis for entropy computation.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_mask = jnp.zeros((n_act,), dtype=jnp.int32)
    else:
        fixed_mask = fixed_joint_mask.astype(jnp.int32)

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)

    # Normalise box inputs to (n_boxes, 3); track whether the call was single-box.
    box_min = jnp.asarray(box_min, dtype=jnp.float32)
    box_max = jnp.asarray(box_max, dtype=jnp.float32)
    batched_input = box_min.ndim == 2
    if not batched_input:
        box_min = box_min[None, :]   # (1, 3)
        box_max = box_max[None, :]   # (1, 3)
    n_boxes = int(box_min.shape[0])

    if not bool(jnp.all(box_max > box_min)):
        raise ValueError("box_max must be strictly greater than box_min for all axes.")

    box_min_np = np.asarray(box_min)   # (n_boxes, 3)
    box_max_np = np.asarray(box_max)   # (n_boxes, 3)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block))
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    # Per-box accumulators.
    cfg_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    ee_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    tgt_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    err_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    collected = [0] * n_boxes
    entropy_done = [False] * n_boxes

    attempts = 0
    if max_batches is None:
        max_batches = max(8, 8 * int(np.ceil(num_samples * n_boxes / seeds_per_launch)))

    carry_cfg = previous_cfg
    key = rng_key

    # Round-robin box assignment: problem p → box (p % n_boxes).
    box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes

    def _all_done() -> bool:
        if target_entropy is not None:
            return all(entropy_done)
        return all(c >= num_samples for c in collected)

    while not _all_done() and attempts < max_batches:
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

        # Per-problem box bounds (round-robin).
        box_mins_pp = box_min[box_idx]   # (n_batch, 3)
        box_maxs_pp = box_max[box_idx]   # (n_batch, 3)

        # Sample init_points per problem from their respective box.
        uniform = jax.random.uniform(key_pts, (n_batch, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)
        init_points_base = box_mins_pp + uniform * (box_maxs_pp - box_mins_pp)
        init_points = jnp.repeat(init_points_base[:, None, :], restarts_per_target, axis=1)

        rng_seed_val = jax.random.randint(
            key_seed,
            (),
            minval=1,
            maxval=np.iinfo(np.int32).max,
            dtype=jnp.int32,
        )

        try:
            best_cfgs, best_ee, best_targets, best_errs, inside = _brownian_motion_batch_select_jit(
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
                box_mins=box_mins_pp,
                box_maxs=box_maxs_pp,
                lower=lower,
                upper=upper,
                fixed_mask=fixed_mask,
                rng_seed=rng_seed_val,
                target_jnt=target_jnt,
                max_iter=max_iter,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
                lambda_init=lambda_init,
                eps_pos=eps_pos,
                noise_std=noise_std,
                n_brownian_steps=n_brownian_steps,
                fk_check_freq=fk_check_freq,
                threads_per_block=threads_per_block,
            )
        except Exception as exc:  # pragma: no cover - runtime/environment dependent
            if "out of memory" in str(exc).lower() and seeds_per_launch > 1:
                seeds_per_launch = max(1, seeds_per_launch // 2)
                box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes
                continue
            raise

        # Collect valid results per box.
        any_valid = False
        for b in range(n_boxes):
            box_b_mask = (box_idx == b) & inside
            valid_b = int(jnp.sum(box_b_mask))

            if verbose:
                pass  # per-box counts reported below

            if valid_b > 0:
                cfg_chunks[b].append(best_cfgs[box_b_mask])
                ee_chunks[b].append(best_ee[box_b_mask])
                tgt_chunks[b].append(best_targets[box_b_mask])
                err_chunks[b].append(best_errs[box_b_mask])
                collected[b] += valid_b
                carry_cfg = best_cfgs[box_b_mask][-1]
                any_valid = True

        if not any_valid:
            carry_cfg = best_cfgs[int(jnp.argmin(best_errs))]

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            total_valid = sum(int(jnp.sum((box_idx == b) & inside)) for b in range(n_boxes))
            print(
                f"  batch {attempts:3d}: {total_valid:4d}/{n_batch} valid "
                f"({total_valid/n_batch*100:.1f}%), "
                f"collected {collected}, "
                f"{_batch_ms:.1f} ms"
            )

        # Per-box entropy-based early stopping.
        if target_entropy is not None:
            for b in range(n_boxes):
                if not entropy_done[b] and len(ee_chunks[b]) > 0:
                    ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks[b]], axis=0)
                    cur_ent = _box_entropy(ee_so_far, box_min_np[b], box_max_np[b], entropy_bins)
                    if cur_ent >= target_entropy:
                        entropy_done[b] = True

    # Warn / raise for boxes that didn't get enough samples.
    if target_entropy is None:
        for b in range(n_boxes):
            if collected[b] < num_samples:
                import warnings
                warnings.warn(
                    f"Box {b}: unable to collect enough in-box IK samples. "
                    f"Collected {collected[b]}/{num_samples} after {attempts} batches. "
                    "Try increasing max_iter/restarts_per_target or widening the box.",
                    stacklevel=2,
                )
        if all(len(cc) == 0 for cc in cfg_chunks):
            raise RuntimeError("No valid in-box samples collected for any box.")

    def _stack(chunks: list[list[Array]]) -> Array:
        return jnp.concatenate(chunks, axis=0)[:num_samples]

    if batched_input:
        cfg_all = jnp.stack([_stack(cfg_chunks[b]) for b in range(n_boxes)])
        ee_all  = jnp.stack([_stack(ee_chunks[b])  for b in range(n_boxes)])
        tgt_all = jnp.stack([_stack(tgt_chunks[b]) for b in range(n_boxes)])
        err_all = jnp.stack([_stack(err_chunks[b]) for b in range(n_boxes)])
    else:
        cfg_all = _stack(cfg_chunks[0])
        ee_all  = _stack(ee_chunks[0])
        tgt_all = _stack(tgt_chunks[0])
        err_all = _stack(err_chunks[0])

    return cfg_all, ee_all, tgt_all, err_all


# ── SVGD sampler ─────────────────────────────────────────────────────────────



def svgd_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
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
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Sample IK configurations whose end-effectors cover one or more box regions using SVGD.

    The CUDA kernel uses SVGD to transport particles and cover the constraint
    manifold uniformly. Each seed is updated using:
      - Gradient of log-target (task-space residuals)
      - RBF kernel repulsion for manifold coverage

    **Batched boxes**: pass ``box_min``/``box_max`` with shape ``(n_boxes, 3)``
    to sample ``num_samples`` configurations for *each* box in a single call.
    Results have a leading ``n_boxes`` dimension; a single-box call (shape
    ``(3,)``) preserves the original ``(num_samples, ...)`` return shape.

    Args:
        robot: Robot model.
        target_link_index: End-effector link index.
        rng_key: JAX PRNG key.
        previous_cfg: Previous joint configuration (warm-start).
        box_min: Box minimum corner(s).  Shape ``(3,)`` or ``(n_boxes, 3)``.
        box_max: Box maximum corner(s).  Matching shape to ``box_min``.
        num_samples: Total number of samples to collect per box.
        seeds_per_launch: Seeds per CUDA kernel launch.
        restarts_per_target: Warm restarts per target point.
        n_iters: Number of SVGD iterations.
        bandwidth: RBF kernel bandwidth.
        step_size: SVGD step size.
        threads_per_block: CUDA threads per block (multiple of 32).
        fixed_joint_mask: Mask for fixed joints.
        memory_limit_gb: Memory budget in GB.
        max_batches: Maximum batches to run.
        target_entropy: If provided, stop when entropy reaches this value (nats).
        entropy_bins: Number of histogram bins per axis.
        verbose: Print timing info.

    Returns:
        Tuple of (cfgs, ee_points, target_points, errors).
        Shapes are ``(num_samples, ...)`` for a single-box call and
        ``(n_boxes, num_samples, ...)`` for a batched call.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_mask = jnp.zeros((n_act,), dtype=jnp.int32)
    else:
        fixed_mask = fixed_joint_mask.astype(jnp.int32)

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)

    # Normalise box inputs to (n_boxes, 3).
    box_min = jnp.asarray(box_min, dtype=jnp.float32)
    box_max = jnp.asarray(box_max, dtype=jnp.float32)
    batched_input = box_min.ndim == 2
    if not batched_input:
        box_min = box_min[None, :]
        box_max = box_max[None, :]
    n_boxes = int(box_min.shape[0])

    if not bool(jnp.all(box_max > box_min)):
        raise ValueError("box_max must be strictly greater than box_min for all axes.")

    box_min_np = np.asarray(box_min)
    box_max_np = np.asarray(box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block))
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    cfg_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    ee_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    tgt_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    err_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    collected = [0] * n_boxes
    entropy_done = [False] * n_boxes

    attempts = 0
    if max_batches is None:
        max_batches = max(
            8, 8 * int(np.ceil(num_samples * n_boxes / (seeds_per_launch * restarts_per_target)))
        )

    carry_cfg = previous_cfg
    key = rng_key

    box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes

    def _all_done() -> bool:
        if target_entropy is not None:
            return all(entropy_done)
        return all(c >= num_samples for c in collected)

    while not _all_done() and attempts < max_batches:
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

        # Per-problem box bounds (round-robin).
        box_mins_pp = box_min[box_idx]   # (n_batch, 3)
        box_maxs_pp = box_max[box_idx]   # (n_batch, 3)

        # Sample init_points per problem from their respective box.
        uniform = jax.random.uniform(key_pts, (n_batch, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)
        init_points_base = box_mins_pp + uniform * (box_maxs_pp - box_mins_pp)
        init_points = jnp.repeat(
            init_points_base[:, None, :], restarts_per_target, axis=1
        )

        rng_seed_val = jax.random.randint(
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
                    box_mins=box_mins_pp,
                    box_maxs=box_maxs_pp,
                    lower=lower,
                    upper=upper,
                    fixed_mask=fixed_mask,
                    rng_seed=rng_seed_val,
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
                box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes
                continue
            raise

        any_valid = False
        for b in range(n_boxes):
            box_b_mask = (box_idx == b) & inside
            valid_b = int(jnp.sum(box_b_mask))

            if valid_b > 0:
                cfg_chunks[b].append(best_cfgs[box_b_mask])
                ee_chunks[b].append(best_ee[box_b_mask])
                tgt_chunks[b].append(best_targets[box_b_mask])
                err_chunks[b].append(best_errs[box_b_mask])
                collected[b] += valid_b
                carry_cfg = best_cfgs[box_b_mask][-1]
                any_valid = True

        if not any_valid:
            carry_cfg = best_cfgs[int(jnp.argmin(best_errs))]

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            total_valid = sum(int(jnp.sum((box_idx == b) & inside)) for b in range(n_boxes))
            print(
                f"  batch {attempts:3d}: {total_valid:4d}/{n_batch} valid "
                f"({total_valid / n_batch * 100:.1f}%), "
                f"collected {collected}, "
                f"{_batch_ms:.1f} ms"
            )

        if target_entropy is not None:
            for b in range(n_boxes):
                if not entropy_done[b] and len(ee_chunks[b]) > 0:
                    ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks[b]], axis=0)
                    cur_ent = _box_entropy(ee_so_far, box_min_np[b], box_max_np[b], entropy_bins)
                    if cur_ent >= target_entropy:
                        entropy_done[b] = True

    if target_entropy is None:
        for b in range(n_boxes):
            if collected[b] < num_samples:
                import warnings
                warnings.warn(
                    f"Box {b}: unable to collect enough in-box IK samples. "
                    f"Collected {collected[b]}/{num_samples} after {attempts} batches. "
                    "Try increasing n_iters/restarts_per_target or widening the box.",
                    stacklevel=2,
                )
        if all(len(cc) == 0 for cc in cfg_chunks):
            raise RuntimeError("No valid in-box samples collected for any box.")

    def _stack(chunks: list[list[Array]]) -> Array:
        return jnp.concatenate(chunks, axis=0)[:num_samples]

    if batched_input:
        cfg_all = jnp.stack([_stack(cfg_chunks[b]) for b in range(n_boxes)])
        ee_all  = jnp.stack([_stack(ee_chunks[b])  for b in range(n_boxes)])
        tgt_all = jnp.stack([_stack(tgt_chunks[b]) for b in range(n_boxes)])
        err_all = jnp.stack([_stack(err_chunks[b]) for b in range(n_boxes)])
    else:
        cfg_all = _stack(cfg_chunks[0])
        ee_all  = _stack(ee_chunks[0])
        tgt_all = _stack(tgt_chunks[0])
        err_all = _stack(err_chunks[0])

    return cfg_all, ee_all, tgt_all, err_all


# ── Hit-and-run sampler ───────────────────────────────────────────────────────

_HIT_AND_RUN_MAX_TPB_BY_SMEM = 384  # MAX_JOINTS=64, MAX_ACT=16 build of CUDA kernel.


@functools.partial(
    jax.jit,
    static_argnames=(
        "target_jnt",
        "max_iter",
        "n_iterations",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "noise_std",
        "threads_per_block",
    ),
)
def _hit_and_run_batch_select_jit(
    seeds: Array,
    twists: Array,
    parent_tf: Array,
    parent_idx: Array,
    act_idx: Array,
    mimic_mul: Array,
    mimic_off: Array,
    mimic_act_idx: Array,
    topo_inv: Array,
    ancestor_mask: Array,
    box_mins: Array,   # (n_problems, 3) — per-problem box mins
    box_maxs: Array,   # (n_problems, 3) — per-problem box maxs
    lower: Array,
    upper: Array,
    fixed_mask: Array,
    rng_seed: Array,
    *,
    target_jnt: int,
    max_iter: int,
    n_iterations: int,
    pos_weight: float,
    ori_weight: float,
    lambda_init: float,
    eps_pos: float,
    eps_ori: float,
    noise_std: float,
    threads_per_block: int = 128,
) -> tuple[Array, Array, Array, Array, Array]:
    from ..cuda_kernels._hit_and_run_ik_cuda import hit_and_run_ik_cuda

    cfgs, errs, ee_points, target_points = hit_and_run_ik_cuda(
        seeds=seeds,
        twists=twists,
        parent_tf=parent_tf,
        parent_idx=parent_idx,
        act_idx=act_idx,
        mimic_mul=mimic_mul,
        mimic_off=mimic_off,
        mimic_act_idx=mimic_act_idx,
        topo_inv=topo_inv,
        ancestor_mask=ancestor_mask,
        box_mins=box_mins,
        box_maxs=box_maxs,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_mask,
        rng_seed=rng_seed,
        target_jnt=target_jnt,
        max_iter=max_iter,
        n_iterations=n_iterations,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        noise_std=noise_std,
        threads_per_block=threads_per_block,
    )

    best_idx = jnp.argmin(errs, axis=1)
    rows = jnp.arange(cfgs.shape[0])
    best_cfgs = cfgs[rows, best_idx]
    best_errs = errs[rows, best_idx]
    best_ee = ee_points[rows, best_idx]
    best_targets = target_points[rows, best_idx]
    inside = jnp.all((best_ee >= box_mins) & (best_ee <= box_maxs), axis=1)
    return best_cfgs, best_ee, best_targets, best_errs, inside


def hit_and_run_sample_box_region_cuda(
    robot: Robot,
    target_link_index: int,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    box_min: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    box_max: Float[Array, "3"] | Float[Array, "n_boxes 3"],
    *,
    num_samples: int = 4096,
    seeds_per_launch: int = 2048,
    restarts_per_target: int = 8,
    max_iter: int = 20,
    n_iterations: int = 100,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    eps_ori: float = 1e-4,
    noise_std: float = 0.02,
    threads_per_block: int = 128,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    memory_limit_gb: float = 2.0,
    max_batches: int | None = None,
    target_entropy: float | None = None,
    entropy_bins: int = 10,
    verbose: bool = False,
) -> (
    Tuple[
        Float[Array, "n_samples n_act"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples 3"],
        Float[Array, "n_samples"],
    ]
    | Tuple[
        Float[Array, "n_boxes n_samples n_act"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples 3"],
        Float[Array, "n_boxes n_samples"],
    ]
):
    """Sample IK configurations whose end-effector points cover one or more box regions.

    The CUDA kernel uses a two-phase strategy per seed:

      Phase 1 – LM Boundary Reach (``max_iter`` iterations):
        Levenberg-Marquardt IK toward a fixed target pre-sampled inside the
        box.  Exits early once the position residual is below ``eps_pos``.

      Phase 2 – Hit-and-Run Shuffle (``n_iterations`` steps):
        Gaussian perturbations in joint space (std = ``noise_std``) clamped to
        joint limits.  Each step proposes a new configuration and accepts it
        based on box feasibility, exploring the in-box region via Markov chain.

    **Batched boxes**: pass ``box_min``/``box_max`` with shape ``(n_boxes, 3)``
    to sample ``num_samples`` configurations for *each* box in a single call.
    Seeds within each launch are distributed round-robin across boxes so all
    regions are explored in parallel.  Results have a leading ``n_boxes``
    dimension; a single-box call (shape ``(3,)``) preserves the original
    ``(num_samples, ...)`` return shape.

    Args:
        box_min: Box minimum corner(s).  Shape ``(3,)`` for a single box or
            ``(n_boxes, 3)`` for multiple boxes.
        box_max: Box maximum corner(s).  Matching shape to ``box_min``.
        max_iter: Maximum LM iterations in Phase 1 (boundary reach).
        n_iterations: Number of hit-and-run steps in Phase 2.
        noise_std: Standard deviation of hit-and-run joint perturbations.
        target_entropy: If provided, sampling stops early once the Shannon
            entropy of the collected EE-point distribution reaches this value
            (nats).  When ``None``, sampling continues until ``num_samples``
            are collected.
        entropy_bins: Histogram bins per axis for entropy computation.
    """
    n_act = int(robot.joints.num_actuated_joints)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_mask = jnp.zeros((n_act,), dtype=jnp.int32)
    else:
        fixed_mask = fixed_joint_mask.astype(jnp.int32)

    target_jnt, ancestor_mask = _compute_ancestor_mask(robot, target_link_index)

    box_min = jnp.asarray(box_min, dtype=jnp.float32)
    box_max = jnp.asarray(box_max, dtype=jnp.float32)
    batched_input = box_min.ndim == 2
    if not batched_input:
        box_min = box_min[None, :]
        box_max = box_max[None, :]
    n_boxes = int(box_min.shape[0])

    if not bool(jnp.all(box_max > box_min)):
        raise ValueError("box_max must be strictly greater than box_min for all axes.")

    box_min_np = np.asarray(box_min)
    box_max_np = np.asarray(box_max)

    if restarts_per_target < 1:
        raise ValueError("restarts_per_target must be >= 1.")
    _validate_threads_per_block(int(threads_per_block), _HIT_AND_RUN_MAX_TPB_BY_SMEM)
    seeds_per_launch = _seeds_per_launch_budget(
        n_act, seeds_per_launch, memory_limit_gb, restarts_per_target
    )

    cfg_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    ee_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    tgt_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    err_chunks: list[list[Array]] = [[] for _ in range(n_boxes)]
    collected = [0] * n_boxes
    entropy_done = [False] * n_boxes

    attempts = 0
    if max_batches is None:
        max_batches = max(8, 8 * int(np.ceil(num_samples * n_boxes / seeds_per_launch)))

    carry_cfg = previous_cfg
    key = rng_key

    box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes

    def _all_done() -> bool:
        if target_entropy is not None:
            return all(entropy_done)
        return all(c >= num_samples for c in collected)

    while not _all_done() and attempts < max_batches:
        attempts += 1
        n_batch = seeds_per_launch

        if verbose:
            import time as _time
            _t_batch = _time.perf_counter()

        key, key_warm, key_rand, key_seed = jax.random.split(key, 4)

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

        box_mins_pp = box_min[box_idx]   # (n_batch, 3)
        box_maxs_pp = box_max[box_idx]   # (n_batch, 3)

        rng_seed = jax.random.randint(
            key_seed, (), minval=1, maxval=np.iinfo(np.int32).max, dtype=jnp.int32,
        )

        try:
            best_cfgs, best_ee, best_targets, best_errs, inside = _hit_and_run_batch_select_jit(
                seeds=seeds,
                twists=robot.joints.twists,
                parent_tf=robot.joints.parent_transforms,
                parent_idx=robot.joints.parent_indices,
                act_idx=robot.joints.actuated_indices,
                mimic_mul=robot.joints.mimic_multiplier,
                mimic_off=robot.joints.mimic_offset,
                mimic_act_idx=robot.joints.mimic_act_indices,
                topo_inv=robot.joints._topo_sort_inv,
                ancestor_mask=ancestor_mask,
                box_mins=box_mins_pp,
                box_maxs=box_maxs_pp,
                lower=lower,
                upper=upper,
                fixed_mask=fixed_mask,
                rng_seed=rng_seed,
                target_jnt=target_jnt,
                max_iter=max_iter,
                n_iterations=n_iterations,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
                lambda_init=lambda_init,
                eps_pos=eps_pos,
                eps_ori=eps_ori,
                noise_std=noise_std,
                threads_per_block=threads_per_block,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower() and seeds_per_launch > 1:
                seeds_per_launch = max(1, seeds_per_launch // 2)
                box_idx = jnp.arange(seeds_per_launch, dtype=jnp.int32) % n_boxes
                continue
            raise

        any_valid = False
        for b in range(n_boxes):
            box_b_mask = (box_idx == b) & inside
            valid_b = int(jnp.sum(box_b_mask))
            if valid_b > 0:
                cfg_chunks[b].append(best_cfgs[box_b_mask])
                ee_chunks[b].append(best_ee[box_b_mask])
                tgt_chunks[b].append(best_targets[box_b_mask])
                err_chunks[b].append(best_errs[box_b_mask])
                collected[b] += valid_b
                carry_cfg = best_cfgs[box_b_mask][-1]
                any_valid = True

        if not any_valid:
            carry_cfg = best_cfgs[int(jnp.argmin(best_errs))]

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            total_valid = sum(int(jnp.sum((box_idx == b) & inside)) for b in range(n_boxes))
            print(
                f"  batch {attempts:3d}: {total_valid:4d}/{n_batch} valid "
                f"({total_valid / n_batch * 100:.1f}%), "
                f"collected {collected}, "
                f"{_batch_ms:.1f} ms"
            )

        if target_entropy is not None:
            for b in range(n_boxes):
                if not entropy_done[b] and len(ee_chunks[b]) > 0:
                    ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks[b]], axis=0)
                    if _box_entropy(ee_so_far, box_min_np[b], box_max_np[b], entropy_bins) >= target_entropy:
                        entropy_done[b] = True

    if target_entropy is None:
        for b in range(n_boxes):
            if collected[b] < num_samples:
                import warnings
                warnings.warn(
                    f"Box {b}: unable to collect enough in-box IK samples. "
                    f"Collected {collected[b]}/{num_samples} after {attempts} batches. "
                    "Try increasing max_iter/restarts_per_target or widening the box.",
                    stacklevel=2,
                )
        if all(len(cc) == 0 for cc in cfg_chunks):
            raise RuntimeError("No valid in-box samples collected for any box.")

    def _stack(chunks: list[list[Array]]) -> Array:
        return jnp.concatenate(chunks, axis=0)[:num_samples]

    if batched_input:
        cfg_all = jnp.stack([_stack(cfg_chunks[b]) for b in range(n_boxes)])
        ee_all  = jnp.stack([_stack(ee_chunks[b])  for b in range(n_boxes)])
        tgt_all = jnp.stack([_stack(tgt_chunks[b]) for b in range(n_boxes)])
        err_all = jnp.stack([_stack(err_chunks[b]) for b in range(n_boxes)])
    else:
        cfg_all = _stack(cfg_chunks[0])
        ee_all  = _stack(ee_chunks[0])
        tgt_all = _stack(tgt_chunks[0])
        err_all = _stack(err_chunks[0])

    return cfg_all, ee_all, tgt_all, err_all
