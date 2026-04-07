"""Region-constrained stochastic IK sampling via CUDA LS-IK FFI."""

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


def _validate_threads_per_block(threads_per_block: int) -> None:
    if threads_per_block < 32 or threads_per_block > 1024 or threads_per_block % 32 != 0:
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
    if threads_per_block > _REGION_IK_MAX_TPB_BY_SMEM:
        raise ValueError(
            f"threads_per_block={threads_per_block} exceeds the current shared-memory "
            f"budget for region_ls_ik_cuda; use <= {_REGION_IK_MAX_TPB_BY_SMEM}."
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
def _region_batch_select_jit(
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
    from ..cuda_kernels._region_ls_ik_cuda import region_ls_ik_cuda

    cfgs, errs, ee_points, target_points = region_ls_ik_cuda(
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
        box_min=box_min,
        box_max=box_max,
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
    inside = jnp.all((best_ee >= box_min[None, :]) & (best_ee <= box_max[None, :]), axis=1)
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


def ls_ik_sample_box_region_cuda(
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
) -> Tuple[
    Float[Array, "n_samples n_act"],
    Float[Array, "n_samples 3"],
    Float[Array, "n_samples 3"],
    Float[Array, "n_samples"],
]:
    """Sample IK configurations whose end-effector points cover a closed box region.

    The CUDA kernel uses a two-phase strategy per seed:

      Phase 1 – LM Boundary Reach (``max_iter`` iterations):
        Levenberg-Marquardt IK toward a fixed target pre-sampled inside the
        box.  Exits early once the position residual is below ``eps_pos``.

      Phase 2 – Brownian Shuffle (``n_brownian_steps`` steps):
        Gaussian perturbations in joint space (std = ``noise_std``) clamped
        to joint limits.  Every ``fk_check_freq`` steps an FK check is run:
        if the EE has drifted outside the box, 2 corrective LM steps push it
        back to the nearest box-boundary point.  Most steps are O(n_act)
        projections, keeping per-thread compute low.

    The returned ``err_all`` contains squared box-distance values (0 when the
    EE is inside the box, positive otherwise), replacing the weighted pose
    cost of the previous kernel.

    Args:
        max_iter: Maximum LM iterations in Phase 1 (boundary reach).
        noise_std: Standard deviation of Brownian joint perturbations in Phase 2.
        n_brownian_steps: Number of Brownian steps in Phase 2.
        fk_check_freq: FK check / corrective-LM period during Phase 2.
            Lower values maintain feasibility more strictly at higher compute cost.
        target_entropy: If provided, sampling stops early once the Shannon entropy
            of the collected EE-point distribution (discretized into entropy_bins^3
            cells) reaches this value (in nats).  Maximum possible entropy is
            log(entropy_bins^3) ≈ 6.91 nats for entropy_bins=10.  When None,
            sampling continues until num_samples are collected.
        entropy_bins: Number of histogram bins per axis for entropy computation.
            Only used when target_entropy is not None.
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

    cfg_chunks: list[Array] = []
    ee_chunks: list[Array] = []
    tgt_chunks: list[Array] = []
    err_chunks: list[Array] = []

    collected = 0
    attempts = 0
    if max_batches is None:
        # Generous retry budget: allow repeated resampling when the requested box
        # includes hard-to-reach regions.
        max_batches = max(8, 8 * int(np.ceil(num_samples / seeds_per_launch)))

    carry_cfg = previous_cfg
    key = rng_key

    while collected < num_samples and attempts < max_batches:
        attempts += 1
        # Always use a full-size batch. Varying n_batch would change the shape
        # of seeds/init_points, causing JAX to retrace _region_batch_select_jit
        # and stalling the warmed-up path. Any over-collection is cheaply
        # sliced away at the end.
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
        init_points = jnp.repeat(init_points_base[:, None, :], restarts_per_target, axis=1)

        rng_seed = jax.random.randint(
            key_seed,
            (),
            minval=1,
            maxval=np.iinfo(np.int32).max,
            dtype=jnp.int32,
        )

        try:
            best_cfgs, best_ee, best_targets, best_errs, inside = _region_batch_select_jit(
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
                continue
            raise

        valid_count = int(jnp.sum(inside))  # forces GPU sync

        if verbose:
            _batch_ms = (_time.perf_counter() - _t_batch) * 1000.0
            print(
                f"  batch {attempts:3d}: {valid_count:4d}/{n_batch} valid "
                f"({valid_count/n_batch*100:.1f}%), "
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

        # Entropy-based early stopping: stop once the collected EE-point
        # distribution is sufficiently diverse inside the box.
        if target_entropy is not None and len(ee_chunks) > 0:
            ee_so_far = np.concatenate([np.asarray(c) for c in ee_chunks], axis=0)
            current_entropy = _box_entropy(ee_so_far, box_min_np, box_max_np, entropy_bins)
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
