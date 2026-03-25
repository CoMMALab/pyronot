"""JAX FFI wrapper for the CUDA forward kinematics kernel.

The companion shared library ``_fk_cuda.so`` must be compiled from
``_fk_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_fk_cuda.sh

Requires JAX >= 0.4.14 (for jax.ffi).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int

_LIB_NAME = "_fk_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"CUDA FK library not found at {lib_path}.\n"
            "Compile it first with:  bash src/pyronot/cuda_kernels/build_fk_cuda.sh\n"
            "(This produces _fk_cuda_lib.so alongside the kernel source.)"
        )
    lib = ctypes.CDLL(str(lib_path))

    # jaxlib requires a PyCapsule (not a raw ctypes function pointer) for
    # api_version=1 (XLA FFI).  Build one from the exported handler symbol.
    # ctypes.pythonapi is an attribute of ctypes, not a separate module.
    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = _PyCapsule_New(
        ctypes.cast(lib.FkCudaFfi, ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("fk_cuda", capsule, platform="CUDA")


def fk_cuda(
    cfg: Float[Array, "*batch n_act"],
    twists: Float[Array, "n_joints 6"],
    parent_tf: Float[Array, "n_joints 7"],
    parent_idx: Int[Array, " n_joints"],
    act_idx: Int[Array, " n_joints"],
    mimic_mul: Float[Array, " n_joints"],
    mimic_off: Float[Array, " n_joints"],
    mimic_act_idx: Int[Array, " n_joints"],
    topo_inv: Int[Array, " n_joints"],
    fk_level_starts: Int[Array, " n_levels_plus_one"],
    fk_level_joints: Int[Array, " n_joints"],
) -> Float[Array, "*batch n_joints 7"]:
    """Run FK on the CUDA kernel via the JAX FFI.

    Produces the same ``(*batch, n_joints, 7)`` [wxyz_xyz] output as the JAX
    implementation, with one CUDA block per batch element and per-depth
    parallelism across joints inside the block.

    Args:
        cfg:          Actuated joint configuration, shape ``(*batch, n_act)``.
        twists:       Per-joint Lie-algebra twist vectors, shape ``(n_joints, 6)``.
        parent_tf:    Constant parent-to-joint transforms, shape ``(n_joints, 7)``.
        parent_idx:   Original parent joint index per joint (-1 for roots).
        act_idx:      Actuated joint source index per joint (-1 if fixed).
        mimic_mul:    Mimic multiplier per joint (1.0 for non-mimic).
        mimic_off:    Mimic offset per joint (0.0 for non-mimic).
        mimic_act_idx: Mimicked actuated index (-1 if not a mimic joint).
        topo_inv:     ``_topo_sort_inv``: maps sorted index i to original index j.
        fk_level_starts: Prefix offsets for FK depth levels.
        fk_level_joints: Joint indices grouped by FK depth level.

    Returns:
        World-frame SE(3) transforms, shape ``(*batch, n_joints, 7)``.
    """
    _load_and_register()

    batch_axes = cfg.shape[:-1]
    batch = int(np.prod(batch_axes)) if batch_axes else 1
    n_act = cfg.shape[-1]
    n_joints = twists.shape[0]

    # Ensure dtypes expected by the kernel.
    cfg_flat      = cfg.reshape(batch, n_act).astype(jnp.float32)
    twists        = twists.astype(jnp.float32)
    parent_tf     = parent_tf.astype(jnp.float32)
    parent_idx    = parent_idx.astype(jnp.int32)
    act_idx       = act_idx.astype(jnp.int32)
    mimic_mul     = mimic_mul.astype(jnp.float32)
    mimic_off     = mimic_off.astype(jnp.float32)
    mimic_act_idx = mimic_act_idx.astype(jnp.int32)
    topo_inv      = topo_inv.astype(jnp.int32)
    fk_level_starts = fk_level_starts.astype(jnp.int32)
    fk_level_joints = fk_level_joints.astype(jnp.int32)

    # In this JAX version ffi_call(target, shape) returns a callable;
    # the inputs are passed when invoking that callable.
    out_flat = jax.ffi.ffi_call(
        "fk_cuda",
        jax.ShapeDtypeStruct((batch, n_joints, 7), jnp.float32),
    )(
        cfg_flat,
        twists,
        parent_tf,
        parent_idx,
        act_idx,
        mimic_mul,
        mimic_off,
        mimic_act_idx,
        topo_inv,
        fk_level_starts,
        fk_level_joints,
    )

    return out_flat.reshape(*batch_axes, n_joints, 7)
