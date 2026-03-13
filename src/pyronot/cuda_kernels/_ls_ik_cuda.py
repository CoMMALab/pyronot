"""JAX FFI wrapper for the CUDA Least Squares IK kernel.

The companion shared library ``_ls_ik_cuda_lib.so`` must be compiled from
``_ls_ik_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh

Provides one primitive:

  ls_ik_cuda — multi-seed LM directly on seeds (no coarse phase).

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

_LIB_NAME = "_ls_ik_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"LS-IK CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New         = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "LsIkCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("ls_ik_cuda", capsule, platform="CUDA")


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


def ls_ik_cuda(
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
    target_jnt:  int,
    max_iter:    int,
    pos_weight:  float,
    ori_weight:  float,
    lambda_init: float,
    eps_pos:     float,
    eps_ori:     float,
) -> tuple[Float[Array, "n_seeds n_act"], Float[Array, "n_seeds"]]:
    """Run multi-seed Levenberg-Marquardt on the GPU.

    One CUDA thread per seed.  Each thread runs ``max_iter`` LM iterations
    with Jacobi column scaling, trust-region step clipping, and a 5-point
    line search.

    Args:
        seeds:         Initial configurations, shape ``(n_seeds, n_act)``.
        twists:        Per-joint twist, shape ``(n_joints, 6)``.
        parent_tf:     Constant parent-to-joint transforms, ``(n_joints, 7)``.
        parent_idx:    Parent joint index per joint (−1 for roots).
        act_idx:       Actuated source index per joint (−1 if fixed).
        mimic_mul:     Mimic multiplier per joint.
        mimic_off:     Mimic offset per joint.
        mimic_act_idx: Mimicked actuated index (−1 if not mimic).
        topo_inv:      Topological sort inverse map.
        ancestor_mask: 1 for joints that are ancestors of the target link joint.
        target_T:      Target end-effector pose ``[w,x,y,z,tx,ty,tz]``.
        lower:         Lower joint limits, shape ``(n_act,)``.
        upper:         Upper joint limits, shape ``(n_act,)``.
        fixed_mask:    1 for actuated joints that should not move.
        target_jnt:    Joint index corresponding to the target link.
        max_iter:      LM iteration budget per seed.
        pos_weight:    Weight on position residual components.
        ori_weight:    Weight on orientation residual components.
        lambda_init:   Initial LM damping factor.
        eps_pos:       Position convergence threshold [m].
        eps_ori:       Orientation convergence threshold [rad].

    Returns:
        Tuple ``(cfgs, errors)`` where ``cfgs`` has shape ``(n_seeds, n_act)``
        and ``errors`` has shape ``(n_seeds,)``.
    """
    _load_and_register()

    n_seeds, n_act = seeds.shape
    seeds = seeds.astype(jnp.float32)
    rb    = _robot_buffers(twists, parent_tf, parent_idx, act_idx,
                           mimic_mul, mimic_off, mimic_act_idx, topo_inv)

    cfgs, errs = jax.ffi.ffi_call(
        "ls_ik_cuda",
        (
            jax.ShapeDtypeStruct((n_seeds, n_act), jnp.float32),
            jax.ShapeDtypeStruct((n_seeds,),       jnp.float32),
        ),
    )(
        seeds,
        *rb,
        ancestor_mask.astype(jnp.int32),
        target_T.astype(jnp.float32),
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        target_jnt  = int(target_jnt),
        max_iter    = int(max_iter),
        pos_weight  = np.float32(pos_weight),
        ori_weight  = np.float32(ori_weight),
        lambda_init = np.float32(lambda_init),
        eps_pos     = np.float32(eps_pos),
        eps_ori     = np.float32(eps_ori),
    )
    return cfgs, errs
