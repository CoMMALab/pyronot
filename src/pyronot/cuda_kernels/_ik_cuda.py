"""JAX FFI wrappers for the CUDA HJCD-IK kernels.

The companion shared library ``_ik_cuda_lib.so`` must be compiled from
``_ik_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_ik_cuda.sh

Provides two primitives called by the CUDA path in ``_hjcd_ik.py``:

  ik_coarse_cuda — Phase 1 greedy coordinate-descent across all seeds.
  ik_lm_cuda     — Phase 2 Levenberg-Marquardt refinement.

Between the two calls Python/JAX handles seed selection (top-K argsort),
perturbation, and winner selection, keeping the kernel interface simple.

Requires JAX >= 0.4.14 (for jax.ffi).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

_LIB_NAME = "_ik_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register both FFI targets (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"CUDA IK library not found at {lib_path}.\n"
            "Compile it first with:  bash src/pyronot/cuda_kernels/build_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    for sym, name in [("IkCoarseCudaFfi", "ik_coarse_cuda"),
                      ("IkLmCudaFfi",     "ik_lm_cuda")]:
        capsule = _PyCapsule_New(
            ctypes.cast(getattr(lib, sym), ctypes.c_void_p),
            b"xla._CUSTOM_CALL_TARGET",
            None,
        )
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")


def _robot_buffers(
    twists:        Float[Array, "n_joints 6"],
    parent_tf:     Float[Array, "n_joints 7"],
    parent_idx:    Int[Array,   " n_joints"],
    act_idx:       Int[Array,   " n_joints"],
    mimic_mul:     Float[Array, " n_joints"],
    mimic_off:     Float[Array, " n_joints"],
    mimic_act_idx: Int[Array,   " n_joints"],
    topo_inv:      Int[Array,   " n_joints"],
) -> tuple:
    """Cast robot-model arrays to the dtypes expected by the C++ kernel."""
    return (
        twists.astype(jnp.float32),
        parent_tf.astype(jnp.float32),
        parent_idx.astype(jnp.int32),
        act_idx.astype(jnp.int32),
        mimic_mul.astype(jnp.float32),
        mimic_off.astype(jnp.float32),
        mimic_act_idx.astype(jnp.int32),
        topo_inv.astype(jnp.int32),
    )


def ik_coarse_cuda(
    seeds:         Float[Array, "n_seeds n_act"],
    twists:        Float[Array, "n_joints 6"],
    parent_tf:     Float[Array, "n_joints 7"],
    parent_idx:    Int[Array,   " n_joints"],
    act_idx:       Int[Array,   " n_joints"],
    mimic_mul:     Float[Array, " n_joints"],
    mimic_off:     Float[Array, " n_joints"],
    mimic_act_idx: Int[Array,   " n_joints"],
    topo_inv:      Int[Array,   " n_joints"],
    ancestor_mask: Int[Array,   " n_joints"],
    target_T:      Float[Array, " 7"],
    lower:         Float[Array, " n_act"],
    upper:         Float[Array, " n_act"],
    fixed_mask:    Int[Array,   " n_act"],
    *,
    target_jnt: int,
    k_max: int,
) -> Float[Array, "n_seeds n_act"]:
    """Run greedy coordinate-descent on all seeds in parallel (Phase 1).

    Args:
        seeds:         Initial configurations, shape ``(n_seeds, n_act)``.
        twists:        Per-joint Lie-algebra twist, shape ``(n_joints, 6)``.
        parent_tf:     Constant parent-to-joint transforms, ``(n_joints, 7)``.
        parent_idx:    Parent joint index per joint (−1 for roots).
        act_idx:       Actuated source index per joint (−1 if fixed).
        mimic_mul:     Mimic multiplier per joint (1.0 for non-mimic).
        mimic_off:     Mimic offset per joint (0.0 for non-mimic).
        mimic_act_idx: Mimicked actuated index (−1 if not mimic).
        topo_inv:      Topological sort inverse map.
        ancestor_mask: 1 for joints that are ancestors of the target link joint.
        target_T:      Target end-effector pose ``[w,x,y,z,tx,ty,tz]``.
        lower:         Lower joint limits, shape ``(n_act,)``.
        upper:         Upper joint limits, shape ``(n_act,)``.
        fixed_mask:    1 for actuated joints that should not move.
        target_jnt:    Joint index corresponding to the target link.
        k_max:         Number of coordinate-descent iterations.

    Returns:
        Refined configurations, shape ``(n_seeds, n_act)``.
    """
    _load_and_register()

    n_seeds, n_act = seeds.shape
    seeds = seeds.astype(jnp.float32)
    rb = _robot_buffers(twists, parent_tf, parent_idx, act_idx,
                        mimic_mul, mimic_off, mimic_act_idx, topo_inv)

    return jax.ffi.ffi_call(
        "ik_coarse_cuda",
        jax.ShapeDtypeStruct((n_seeds, n_act), jnp.float32),
    )(
        seeds,
        *rb,
        ancestor_mask.astype(jnp.int32),
        target_T.astype(jnp.float32),
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        target_jnt=int(target_jnt),
        k_max=int(k_max),
    )


def ik_lm_cuda(
    seeds:         Float[Array, "n_seeds n_act"],
    noise:         Float[Array, "n_seeds max_iter n_act"],
    twists:        Float[Array, "n_joints 6"],
    parent_tf:     Float[Array, "n_joints 7"],
    parent_idx:    Int[Array,   " n_joints"],
    act_idx:       Int[Array,   " n_joints"],
    mimic_mul:     Float[Array, " n_joints"],
    mimic_off:     Float[Array, " n_joints"],
    mimic_act_idx: Int[Array,   " n_joints"],
    topo_inv:      Int[Array,   " n_joints"],
    ancestor_mask: Int[Array,   " n_joints"],
    target_T:      Float[Array, " 7"],
    lower:         Float[Array, " n_act"],
    upper:         Float[Array, " n_act"],
    fixed_mask:    Int[Array,   " n_act"],
    *,
    target_jnt: int,
    max_iter: int,
    stall_patience: int,
    lambda_init: float,
    limit_prior_weight: float,
    kick_scale: float,
    eps_pos: float,
    eps_ori: float,
) -> Float[Array, "n_seeds n_act"]:
    """Run Levenberg-Marquardt refinement on all seeds in parallel (Phase 2).

    Args:
        seeds:               Initial configurations, shape ``(n_seeds, n_act)``.
        noise:               Pre-generated Gaussian kick noise,
                             shape ``(n_seeds, max_iter, n_act)``.
        twists, …, topo_inv: Robot model arrays (same as ik_coarse_cuda).
        ancestor_mask:       Ancestor mask (same as ik_coarse_cuda).
        target_T:            Target pose ``[w,x,y,z,tx,ty,tz]``.
        lower / upper:       Joint limits.
        fixed_mask:          Fixed-joint mask.
        target_jnt:          Joint index for end-effector.
        max_iter:            LM iteration budget.
        stall_patience:      Consecutive non-improving steps before a kick.
        lambda_init:         Initial LM damping factor.
        limit_prior_weight:  Strength of soft joint-limit prior.
        kick_scale:          Standard deviation of random kick.
        eps_pos:             Position convergence threshold [m].
        eps_ori:             Orientation convergence threshold [rad].

    Returns:
        Best-seen configurations for each seed, shape ``(n_seeds, n_act)``.
    """
    _load_and_register()

    n_seeds, n_act = seeds.shape
    seeds = seeds.astype(jnp.float32)
    noise = noise.astype(jnp.float32)
    rb = _robot_buffers(twists, parent_tf, parent_idx, act_idx,
                        mimic_mul, mimic_off, mimic_act_idx, topo_inv)

    return jax.ffi.ffi_call(
        "ik_lm_cuda",
        jax.ShapeDtypeStruct((n_seeds, n_act), jnp.float32),
    )(
        seeds,
        noise,
        *rb,
        ancestor_mask.astype(jnp.int32),
        target_T.astype(jnp.float32),
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        target_jnt=int(target_jnt),
        max_iter=int(max_iter),
        stall_patience=int(stall_patience),
        lambda_init=np.float32(lambda_init),
        limit_prior_weight=np.float32(limit_prior_weight),
        kick_scale=np.float32(kick_scale),
        eps_pos=np.float32(eps_pos),
        eps_ori=np.float32(eps_ori),
    )
