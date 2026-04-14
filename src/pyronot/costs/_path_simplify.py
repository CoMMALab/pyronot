from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _active_order(keep: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return active waypoint order and active counts for each batch item."""
    batch_size, max_waypoints = keep.shape
    abs_idx = jnp.broadcast_to(jnp.arange(max_waypoints, dtype=jnp.int32), (batch_size, max_waypoints))
    sort_key = jnp.where(keep, abs_idx, abs_idx + max_waypoints)
    order = jnp.argsort(sort_key, axis=1)
    counts = jnp.sum(keep, axis=1, dtype=jnp.int32)
    return order, counts


def simplify_paths_batched(
    paths: np.ndarray,
    simplify_mask: np.ndarray,
    edge_validator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_shortcut_rounds: int = 64,
    min_improvement: float = 1e-9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched OMPL-style randomized shortcut path simplification.

    This mirrors the common OMPL simplification strategy of repeatedly trying
    random non-adjacent waypoint shortcuts and accepting a shortcut only if it
    is both valid and shortens the current polyline.

    Args:
        paths: Array of shape (B, T, D).
        simplify_mask: Boolean array of shape (B,). Paths with False are left unchanged.
        edge_validator: Callable that checks direct motion validity for endpoint
            pairs. It receives (N, D) and (N, D) arrays and returns (N,) bool.
        num_shortcut_rounds: Number of randomized shortcut rounds.
        min_improvement: Minimum required shortening to accept a shortcut.
        seed: Optional RNG seed.

    Returns:
        simplified_paths: Array of shape (B, T, D).
        waypoint_counts: Number of active waypoints per simplified path, shape (B,).
    """
    if paths.ndim != 3:
        raise ValueError(f"paths must be rank-3 (B, T, D), got shape={paths.shape}")

    batch_size, max_waypoints, _ = paths.shape
    if simplify_mask.shape != (batch_size,):
        raise ValueError(
            f"simplify_mask must have shape ({batch_size},), got {simplify_mask.shape}"
        )

    key = jax.random.PRNGKey(0 if seed is None else int(seed))

    paths_j = jnp.asarray(paths)
    simplify_mask_j = jnp.asarray(simplify_mask, dtype=bool)

    # Keep mask over original waypoint slots; simplify by dropping intermediates.
    keep = jnp.ones((batch_size, max_waypoints), dtype=bool)
    # Always preserve endpoints.
    keep = keep.at[:, 0].set(True)
    keep = keep.at[:, max_waypoints - 1].set(True)

    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)

    rounds = max(0, int(num_shortcut_rounds))
    for _ in range(rounds):
        key, k_i, k_j = jax.random.split(key, 3)

        active_idx, active_counts = _active_order(keep)
        can_try = jnp.logical_and(simplify_mask_j, active_counts > 2)

        # Sample shortcut positions in active-order space (not absolute index space).
        span_i = jnp.maximum(active_counts - 2, 1)
        pos_i = jax.random.randint(k_i, (batch_size,), minval=0, maxval=span_i)
        pos_i = jnp.where(can_try, pos_i, 0)

        span_j = jnp.maximum(active_counts - (pos_i + 2), 1)
        off_j = jax.random.randint(k_j, (batch_size,), minval=0, maxval=span_j)
        pos_j = pos_i + 2 + off_j
        pos_j = jnp.minimum(pos_j, active_counts - 1)

        abs_i = active_idx[batch_idx, pos_i]
        abs_j = active_idx[batch_idx, pos_j]

        pts_active = jnp.take_along_axis(paths_j, active_idx[..., None], axis=1)
        seg = jnp.linalg.norm(jnp.diff(pts_active, axis=1), axis=-1)
        seg_valid = jnp.arange(max_waypoints - 1, dtype=jnp.int32)[None, :] < (active_counts[:, None] - 1)
        seg = jnp.where(seg_valid, seg, 0.0)

        # Prefix with zero to query subpath length in O(1).
        pref = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=seg.dtype), jnp.cumsum(seg, axis=1)], axis=1)
        old_len = pref[batch_idx, pos_j] - pref[batch_idx, pos_i]
        new_len = jnp.linalg.norm(paths_j[batch_idx, abs_i, :] - paths_j[batch_idx, abs_j, :], axis=-1)
        improves = new_len + min_improvement < old_len

        # Validate candidate direct edges in one batched call.
        a_np = np.asarray(paths_j[batch_idx, abs_i, :])
        b_np = np.asarray(paths_j[batch_idx, abs_j, :])
        valid_np = np.asarray(edge_validator(a_np, b_np), dtype=bool)
        valid = jnp.asarray(valid_np)

        accept = jnp.logical_and(can_try, jnp.logical_and(improves, valid))

        # Map absolute waypoint -> active rank, then remove active ranks in (i, j).
        rank_by_abs = jnp.zeros((batch_size, max_waypoints), dtype=jnp.int32)
        rank_by_abs = rank_by_abs.at[batch_idx[:, None], active_idx].set(
            jnp.broadcast_to(jnp.arange(max_waypoints, dtype=jnp.int32), (batch_size, max_waypoints))
        )

        remove_mid = jnp.logical_and(
            rank_by_abs > pos_i[:, None],
            rank_by_abs < pos_j[:, None],
        )
        remove_mid = jnp.logical_and(remove_mid, keep)
        remove_mid = jnp.logical_and(remove_mid, accept[:, None])

        keep = jnp.logical_and(keep, jnp.logical_not(remove_mid))
        keep = keep.at[:, 0].set(True)
        keep = keep.at[:, max_waypoints - 1].set(True)

    final_idx, final_counts = _active_order(keep)
    compact = jnp.take_along_axis(paths_j, final_idx[..., None], axis=1)

    last_abs = final_idx[batch_idx, final_counts - 1]
    last_cfg = paths_j[batch_idx, last_abs, :]
    padded = jnp.tile(last_cfg[:, None, :], (1, max_waypoints, 1))
    active_pos = jnp.arange(max_waypoints, dtype=jnp.int32)[None, :] < final_counts[:, None]
    padded = jnp.where(active_pos[..., None], compact, padded)

    return np.asarray(padded), np.asarray(final_counts, dtype=np.int32)
