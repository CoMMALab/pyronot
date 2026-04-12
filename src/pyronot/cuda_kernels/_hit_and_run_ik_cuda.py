"""JAX FFI wrapper for hit-and-run region IK CUDA kernel.

Compile the companion shared library first:

    bash src/pyronot/cuda_kernels/build_hit_and_run_ik_cuda.sh
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

_LIB_NAME = "_hit_and_run_ik_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"Hit-and-run IK CUDA library not found at {lib_path}.\n"
            "Compile it first with: bash src/pyronot/cuda_kernels/build_hit_and_run_ik_cuda.sh\n"
        )

    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "HitAndRunIkCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("hit_and_run_ik_cuda", capsule, platform="CUDA")


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


def hit_and_run_ik_cuda(
    seeds: Float[Array, "n_problems n_samples n_act "],
    twists: Float[Array, "n_joints 6"],
    parent_tf: Float[Array, "n_joints 7"],
    parent_idx: Int[Array, " n_joints"],
    act_idx: Int[Array, " n_joints"],
    mimic_mul: Float[Array, " n_joints"],
    mimic_off: Float[Array, " n_joints"],
    mimic_act_idx: Int[Array, " n_joints"],
    topo_inv: Int[Array, " n_joints"],
    ancestor_mask: Int[Array, " n_joints"],
    box_mins: Float[Array, "n_problems 3"],
    box_maxs: Float[Array, "n_problems 3"],
    lower: Float[Array, " n_act"],
    upper: Float[Array, " n_act"],
    fixed_mask: Int[Array, " n_act"],
    rng_seed: Int[Array, ""],
    *,
    target_jnt: int,
    max_iter: int,
    n_iterations: int,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    lambda_init: float = 5e-3,
    eps_pos: float = 1e-4,
    eps_ori: float = 1e-4,
    noise_std: float = 0.01,
    threads_per_block: int = 128,
) -> tuple[
    Float[Array, "n_problems n_samples n_act"],
    Float[Array, "n_problems n_samples"],
    Float[Array, "n_problems n_samples 3"],
    Float[Array, "n_problems n_samples 3"],
]:
    """Run multi-seed hit-and-run region IK on the GPU.

    Each problem has its own axis-aligned box defined by box_mins[p] and
    box_maxs[p], enabling multiple distinct regions in a single kernel launch.
    """
    _load_and_register()

    n_problems, n_samples, n_act = seeds.shape
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

    cfgs, errs, ee_points, targets = jax.ffi.ffi_call(
        "hit_and_run_ik_cuda",
        (
            jax.ShapeDtypeStruct((n_problems, n_samples, n_act), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_samples), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_samples, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_samples, 3), jnp.float32),
        ),
    )(
        seeds.astype(jnp.float32),
        *rb,
        ancestor_mask.astype(jnp.int32),
        box_mins.astype(jnp.float32),
        box_maxs.astype(jnp.float32),
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        rng_seed.astype(jnp.int32),
        target_jnt=int(target_jnt),
        max_iter=int(max_iter),
        n_iterations=int(n_iterations),
        pos_weight=np.float32(pos_weight),
        ori_weight=np.float32(ori_weight),
        lambda_init=np.float32(lambda_init),
        eps_pos=np.float32(eps_pos),
        eps_ori=np.float32(eps_ori),
        noise_std=np.float32(noise_std),
    )

    return cfgs, errs, ee_points, targets
