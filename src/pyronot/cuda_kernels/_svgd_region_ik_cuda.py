"""JAX FFI wrapper for the CUDA SVGD region IK kernel.

The companion shared library ``_svgd_region_ik_cuda_lib.so`` must be compiled from
``_svgd_region_ik_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_svgd_region_ik_cuda.sh

Provides one primitive:

  svgd_region_ik_cuda — multi-seed SVGD region-based inverse kinematics.

Requires JAX >= 0.4.14 (for jax.ffi).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int

_LIB_NAME = "_svgd_region_ik_cuda_lib.so"


def _robot_buffers(
    twists: Float[Array, "n_joints 6"],
    parent_tf: Float[Array, "n_joints 7"],
    parent_idx: Int[Array, " n_joints"],
    act_idx: Int[Array, " n_joints"],
    mimic_mul: Float[Array, " n_joints"],
    mimic_off: Float[Array, " n_joints"],
    mimic_act_idx: Int[Array, " n_joints"],
    topo_inv: Int[Array, " n_joints"],
) -> tuple:
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


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"SVGD region-IK CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash src/pyronot/cuda_kernels/build_svgd_region_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "SvgdRegionIkCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("svgd_region_ik_cuda", capsule, platform="CUDA")


def svgd_region_ik_cuda(
    seeds: Float[Array, "n_problems n_particles n_act"],
    init_points: Float[Array, "n_problems n_particles 3"],
    twists: Float[Array, "n_joints 6"],
    parent_tf: Float[Array, "n_joints 7"],
    parent_idx: Int[Array, " n_joints"],
    act_idx: Int[Array, " n_joints"],
    mimic_mul: Float[Array, " n_joints"],
    mimic_off: Float[Array, " n_joints"],
    mimic_act_idx: Int[Array, " n_joints"],
    topo_inv: Int[Array, " n_joints"],
    target_jnts: Int[Array, "n_ee"],
    ancestor_masks: Int[Array, "n_ee n_joints"],
    lower: Float[Array, " n_act"],
    upper: Float[Array, " n_act"],
    fixed_mask: Int[Array, " n_act"],
    *,
    n_iters: int,
    bandwidth: float,
    step_size: float,
) -> tuple[
    Float[Array, "n_problems n_particles n_act"],
    Float[Array, "n_problems n_particles"],
    Float[Array, "n_problems n_particles 3"],
    Float[Array, "n_problems n_particles 3"],
]:
    """Run multi-seed SVGD region-based IK on the GPU.

    Uses Stein Variational Gradient Descent with RBF kernel repulsion to cover
    the kinematic constraint manifold uniformly. The algorithm transports particles
    to minimize task-space residuals while maintaining spread through kernel-based
    repulsion.

    ``init_points`` provides the per-problem target positions (all particles in a
    problem share the same target, taken from ``init_points[:, 0, :]``).  An
    identity rotation is paired with each position to build the 7-D target
    transform required by the kernel.
    """
    _load_and_register()

    n_problems, n_particles, n_act = seeds.shape
    seeds = seeds.astype(jnp.float32)

    # Build per-problem target transforms [qw, qx, qy, qz, tx, ty, tz] from
    # the target positions in init_points.  All particles in a problem share
    # the same target, so we take the first particle's position.
    target_pos = init_points[:, 0, :].astype(jnp.float32)  # (n_problems, 3)
    identity_quat = jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (n_problems, 4)
    )
    # target_T shape: (n_problems, 1, 7) — one EE per problem
    target_T = jnp.concatenate([identity_quat, target_pos], axis=1)[:, None, :]

    rb = _robot_buffers(
        twists,
        parent_tf,
        parent_idx,
        act_idx,
        mimic_mul,
        mimic_off,
        mimic_act_idx,
        topo_inv,
    )

    cfgs, errs, ee_points, target_points = jax.ffi.ffi_call(
        "svgd_region_ik_cuda",
        (
            jax.ShapeDtypeStruct((n_problems, n_particles, n_act), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_particles), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_particles, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_particles, 3), jnp.float32),
        ),
    )(
        seeds,
        *rb,
        target_jnts.astype(jnp.int32),
        ancestor_masks.astype(jnp.int32),
        target_T,
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        n_iters=int(n_iters),
        bandwidth=np.float32(bandwidth),
        step_size=np.float32(step_size),
    )
    return cfgs, errs, ee_points, target_points
