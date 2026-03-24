"""JAX FFI wrapper for the CUDA STOMP/MPPI TrajOpt kernel.

The companion shared library ``_stomp_trajopt_cuda_lib.so`` must be compiled
from ``_stomp_trajopt_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_stomp_trajopt_cuda.sh

Provides:

  stomp_trajopt_cuda — full STOMP/MPPI trajectory optimisation on the GPU.
      3-kernel-per-iteration architecture (eval, softmax, update).
      Perturbations are generated inline via FIR RNG replay — no global
      noise buffer, no L_inv_T matrix multiply.

Requires JAX >= 0.4.14 (for jax.ffi).
Requires a RobotCollisionSpherized collision model (sphere-based).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

if TYPE_CHECKING:
    from pyronot._robot import Robot
    from pyronot.collision._robot_collision import RobotCollisionSpherized
    from pyronot.optimization_engines._stomp_optimization import StompTrajOptConfig

_LIB_NAME = "_stomp_trajopt_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"STOMP TrajOpt CUDA library not found at {lib_path}.\n"
            "Compile it first with:\n"
            "  bash src/pyronot/cuda_kernels/build_stomp_trajopt_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New         = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "StompTrajoptCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("stomp_trajopt_cuda", capsule, platform="CUDA")


# ---------------------------------------------------------------------------
# World geometry extraction  (identical to SCO/CHOMP wrappers)
# ---------------------------------------------------------------------------

def _extract_world_arrays(
    world_geoms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pyronot.collision._geometry import Sphere, Capsule, Box, HalfSpace

    empty_s = np.zeros((0, 4),  dtype=np.float32)
    empty_c = np.zeros((0, 7),  dtype=np.float32)
    empty_b = np.zeros((0, 15), dtype=np.float32)
    empty_h = np.zeros((0, 6),  dtype=np.float32)

    sph_list, cap_list, box_list, hs_list = [], [], [], []

    if not hasattr(world_geoms, "__iter__"):
        world_geoms = (world_geoms,)

    for wg in world_geoms:
        axes = wg.get_batch_axes()
        if len(axes) == 0:
            wg = wg.broadcast_to((1,))
            axes = (1,)
        if len(axes) > 1:
            wg = wg.reshape((-1,))

        if isinstance(wg, Sphere):
            centers = np.asarray(wg.pose.translation(), dtype=np.float32)
            radii   = np.asarray(wg.radius,             dtype=np.float32)
            sph_list.append(np.concatenate([centers, radii[:, None]], axis=-1))
        elif isinstance(wg, Capsule):
            centers = np.asarray(wg.pose.translation(), dtype=np.float32)
            axes_v  = np.asarray(wg.axis,               dtype=np.float32)
            heights = np.asarray(wg.height,             dtype=np.float32)
            radii   = np.asarray(wg.radius,             dtype=np.float32)
            half_h  = heights[:, None] * 0.5
            a       = centers - axes_v * half_h
            b_      = centers + axes_v * half_h
            cap_list.append(np.concatenate([a, b_, radii[:, None]], axis=-1))
        elif isinstance(wg, Box):
            centers = np.asarray(wg.pose.translation(),          dtype=np.float32)
            R       = np.asarray(wg.pose.rotation().as_matrix(), dtype=np.float32)
            ax1     = R[..., :, 0]
            ax2     = R[..., :, 1]
            ax3     = R[..., :, 2]
            hl      = np.asarray(wg.half_lengths, dtype=np.float32)
            box_list.append(np.concatenate([centers, ax1, ax2, ax3, hl], axis=-1))
        elif isinstance(wg, HalfSpace):
            normals = np.asarray(
                wg.pose.rotation().as_matrix()[..., :, 2], dtype=np.float32)
            points  = np.asarray(wg.pose.translation(), dtype=np.float32)
            hs_list.append(np.concatenate([normals, points], axis=-1))
        else:
            raise NotImplementedError(
                f"stomp_trajopt_cuda: unsupported world geometry type "
                f"{type(wg).__name__}. Supported: Sphere, Capsule, Box, HalfSpace."
            )

    spheres    = np.concatenate(sph_list,  axis=0) if sph_list  else empty_s
    capsules   = np.concatenate(cap_list,  axis=0) if cap_list  else empty_c
    boxes      = np.concatenate(box_list,  axis=0) if box_list  else empty_b
    halfspaces = np.concatenate(hs_list,   axis=0) if hs_list   else empty_h
    return spheres, capsules, boxes, halfspaces


# ---------------------------------------------------------------------------
# Main Python entry point
# ---------------------------------------------------------------------------

def stomp_trajopt_cuda(
    init_trajs:  Float[Array, "B T n_act"],
    start:       Float[Array, " n_act"],
    goal:        Float[Array, " n_act"],
    robot:       "Robot",
    robot_coll:  "RobotCollisionSpherized",
    world_geoms: tuple,
    opt_cfg:     "StompTrajOptConfig",
    *,
    key:         Array | None = None,
) -> tuple[Float[Array, "T n_act"], Float[Array, "B"], Float[Array, "B T n_act"]]:
    """CUDA-accelerated STOMP/MPPI trajectory optimisation.

    Drop-in replacement for ``stomp_trajopt()`` that runs the full STOMP
    loop — FK, collision cost, importance weighting — in a single tightly-
    coupled CUDA kernel (one block per batch trajectory, K threads per block).

    Args:
        init_trajs:  Initial trajectory batch, shape ``[B, T, n_act]``.
        start:       Start joint configuration, shape ``[n_act]``.
        goal:        Goal joint configuration, shape ``[n_act]``.
        robot:       Robot kinematics pytree (``Robot``).
        robot_coll:  Sphere-based collision model (``RobotCollisionSpherized``).
        world_geoms: Tuple of world collision geometry objects.
        opt_cfg:     STOMP hyperparameters (``StompTrajOptConfig``).
        key:         Optional JAX PRNG key (used to seed CUDA RNG).

    Returns:
        best_traj:   Trajectory with lowest final nonlinear cost, ``[T, n_act]``.
        costs:       Final nonlinear cost per trajectory, ``[B]``.
        final_trajs: All optimised trajectories, ``[B, T, n_act]``.

    Raises:
        RuntimeError: If the compiled library is not found.
        TypeError:    If ``robot_coll`` is not a ``RobotCollisionSpherized``.
    """
    from pyronot.collision._robot_collision import RobotCollisionSpherized

    if not isinstance(robot_coll, RobotCollisionSpherized):
        raise TypeError(
            "stomp_trajopt_cuda requires a RobotCollisionSpherized collision model. "
            f"Got: {type(robot_coll).__name__}"
        )

    _load_and_register()

    B, T, n_act = init_trajs.shape
    n_joints    = robot.joints.num_joints

    # ── Robot FK parameters ────────────────────────────────────────────────
    twists        = jnp.asarray(robot.joints.twists,           dtype=jnp.float32)
    parent_tf     = jnp.asarray(robot.joints.parent_transforms, dtype=jnp.float32)
    parent_idx    = jnp.asarray(robot.joints.parent_indices,   dtype=jnp.int32)
    act_idx       = jnp.asarray(robot.joints.actuated_indices, dtype=jnp.int32)
    mimic_mul     = jnp.asarray(robot.joints.mimic_multiplier, dtype=jnp.float32)
    mimic_off     = jnp.asarray(robot.joints.mimic_offset,     dtype=jnp.float32)
    mimic_act_idx = jnp.asarray(robot.joints.mimic_act_indices, dtype=jnp.int32)
    topo_inv      = jnp.asarray(robot.joints._topo_sort_inv,   dtype=jnp.int32)

    lower = jnp.asarray(robot.joints.lower_limits, dtype=jnp.float32)
    upper = jnp.asarray(robot.joints.upper_limits, dtype=jnp.float32)

    # ── Sphere collision model ─────────────────────────────────────────────
    sphere_off_np = np.asarray(robot_coll.coll.pose.translation(), dtype=np.float32)
    sphere_rad_np = np.asarray(robot_coll.coll.radius,             dtype=np.float32)
    N, S = sphere_off_np.shape[:2]

    sphere_offsets = jnp.asarray(sphere_off_np.reshape(-1), dtype=jnp.float32)
    sphere_radii   = jnp.asarray(sphere_rad_np.reshape(-1), dtype=jnp.float32)

    pair_i = jnp.asarray(robot_coll.active_idx_i, dtype=jnp.int32)
    pair_j = jnp.asarray(robot_coll.active_idx_j, dtype=jnp.int32)

    # ── World geometry ─────────────────────────────────────────────────────
    ws_np, wc_np, wb_np, wh_np = _extract_world_arrays(world_geoms)
    world_spheres    = jnp.asarray(ws_np)
    world_capsules   = jnp.asarray(wc_np)
    world_boxes      = jnp.asarray(wb_np)
    world_halfspaces = jnp.asarray(wh_np)

    # ── Endpoints ──────────────────────────────────────────────────────────
    start_f = jnp.asarray(start, dtype=jnp.float32)
    goal_f  = jnp.asarray(goal,  dtype=jnp.float32)

    init_pinned = jnp.asarray(init_trajs, dtype=jnp.float32)
    init_pinned = init_pinned.at[:, 0, :].set(start_f)
    init_pinned = init_pinned.at[:, -1, :].set(goal_f)

    # ── RNG seed from JAX key ──────────────────────────────────────────────
    if key is None:
        key = jax.random.PRNGKey(0)
    rng_seed = int(jax.random.randint(key, (), 0, 2**31 - 1))

    # ── Workspace for multi-kernel STOMP ─────────────────────────────────
    # Layout: best_trajs[B*T*n_act] | best_costs[B] | costs[B*K] | weights[B*K]
    # No noise buffer — perturbations are regenerated inline via FIR RNG replay.
    K = opt_cfg.n_samples
    workspace_size = B * T * n_act + B + 2 * B * K

    # ── FFI call ───────────────────────────────────────────────────────────
    out_trajs, out_costs, _ = jax.ffi.ffi_call(
        "stomp_trajopt_cuda",
        (
            jax.ShapeDtypeStruct((B, T, n_act),      jnp.float32),
            jax.ShapeDtypeStruct((B,),               jnp.float32),
            jax.ShapeDtypeStruct((workspace_size,),  jnp.float32),
        ),
    )(
        init_pinned,
        twists, parent_tf, parent_idx, act_idx,
        mimic_mul, mimic_off, mimic_act_idx, topo_inv,
        sphere_offsets, sphere_radii,
        pair_i, pair_j,
        world_spheres, world_capsules, world_boxes, world_halfspaces,
        lower, upper, start_f, goal_f,
        n_iters                 = np.int64(opt_cfg.n_iters),
        n_samples               = np.int64(opt_cfg.n_samples),
        S                       = np.int64(S),
        noise_scale             = np.float32(opt_cfg.noise_scale),
        temperature             = np.float32(opt_cfg.temperature),
        step_size               = np.float32(opt_cfg.step_size),
        w_smooth                = np.float32(opt_cfg.w_smooth),
        w_acc                   = np.float32(opt_cfg.w_acc),
        w_jerk                  = np.float32(opt_cfg.w_jerk),
        w_limits                = np.float32(opt_cfg.w_limits),
        w_collision             = np.float32(opt_cfg.w_collision),
        w_collision_max         = np.float32(opt_cfg.w_collision_max),
        collision_penalty_scale = np.float32(opt_cfg.collision_penalty_scale),
        collision_margin        = np.float32(opt_cfg.collision_margin),
        rng_seed                = np.int64(rng_seed),
    )

    best_idx  = jnp.argmin(out_costs)
    best_traj = out_trajs[best_idx]

    return best_traj, out_costs, out_trajs
