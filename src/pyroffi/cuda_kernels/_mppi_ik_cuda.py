"""JAX FFI wrapper for the CUDA MPPI+L-BFGS IK kernel.

The companion shared library ``_mppi_ik_cuda_lib.so`` must be compiled from
``_mppi_ik_cuda_kernel.cu`` before this module can be imported:

    bash src/pyroffi/cuda_kernels/build_mppi_ik_cuda.sh

Provides one primitive:

  mppi_ik_cuda — multi-seed MPPI coarse search followed by L-BFGS refinement.

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

_LIB_NAME = "_mppi_ik_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register the FFI target (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"MPPI-IK CUDA library not found at {lib_path}.\n"
            "Compile it first with:  bash src/pyroffi/cuda_kernels/build_mppi_ik_cuda.sh\n"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New         = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = _PyCapsule_New(
        ctypes.cast(getattr(lib, "MppiIkCudaFfi"), ctypes.c_void_p),
        b"xla._CUSTOM_CALL_TARGET",
        None,
    )
    jax.ffi.register_ffi_target("mppi_ik_cuda", capsule, platform="CUDA")


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


def mppi_ik_cuda(
    seeds:          Float[Array, "n_problems n_seeds n_act"],
    twists:         Float[Array, "n_joints 6"],
    parent_tf:      Float[Array, "n_joints 7"],
    parent_idx:     Int[Array,   " n_joints"],
    act_idx:        Int[Array,   " n_joints"],
    mimic_mul:      Float[Array, " n_joints"],
    mimic_off:      Float[Array, " n_joints"],
    mimic_act_idx:  Int[Array,   " n_joints"],
    topo_inv:       Int[Array,   " n_joints"],
    target_jnts:    Int[Array,   "n_ee"],
    ancestor_masks: Int[Array,   "n_ee n_joints"],
    target_T:       Float[Array, "n_problems n_ee 7"],
    robot_spheres_local: Float[Array, "n_rs 4"],
    robot_sphere_joint_idx: Int[Array, " n_rs"],
    world_spheres:  Float[Array, "n_ws 4"],
    world_capsules: Float[Array, "n_wc 7"],
    world_boxes:    Float[Array, "n_wb 15"],
    world_halfspaces: Float[Array, "n_wh 6"],
    lower:          Float[Array, " n_act"],
    upper:          Float[Array, " n_act"],
    fixed_mask:     Int[Array,   " n_act"],
    rng_seed:       Int[Array,   ""],
    *,
    n_particles:      int,
    n_mppi_iters:     int,
    n_lbfgs_iters:    int,
    m_lbfgs:          int,
    sigma:            float,
    mppi_temperature: float,
    pos_weight:       float,
    ori_weight:       float,
    eps_pos:          float,
    eps_ori:          float,
    enable_collision: bool,
    collision_weight: float,
    collision_margin: float,
) -> tuple[Float[Array, "n_problems n_seeds n_act"], Float[Array, "n_problems n_seeds"]]:
    """Run multi-seed MPPI+L-BFGS IK on the GPU.

    Stage 1 (MPPI): ``n_mppi_iters`` particle-based stochastic updates, each
    sampling ``n_particles`` Gaussian perturbations and computing a
    temperature-weighted mean update.

    Stage 2 (L-BFGS): ``n_lbfgs_iters`` quasi-Newton gradient steps using
    the limited-memory BFGS Hessian approximation (``m_lbfgs`` history pairs)
    with 5-point line search and trust-region clipping.

    Args:
        seeds:            Initial configurations, shape ``(n_problems, n_seeds, n_act)``.
        twists:           Per-joint twist, shape ``(n_joints, 6)``.
        parent_tf:        Constant parent-to-joint transforms, ``(n_joints, 7)``.
        parent_idx:       Parent joint index per joint (−1 for roots).
        act_idx:          Actuated source index per joint (−1 if fixed).
        mimic_mul:        Mimic multiplier per joint.
        mimic_off:        Mimic offset per joint.
        mimic_act_idx:    Mimicked actuated index (−1 if not mimic).
        topo_inv:         Topological sort inverse map.
        target_jnts:      Joint index per EE, shape ``(n_ee,)``.
        ancestor_masks:   Ancestor bitmask per EE, shape ``(n_ee, n_joints)``.
        target_T:         Target poses, shape ``(n_problems, n_ee, 7)``.
        lower:            Lower joint limits, shape ``(n_act,)``.
        upper:            Upper joint limits, shape ``(n_act,)``.
        fixed_mask:       1 for actuated joints that should not move.
        n_particles:      Particles per MPPI step (≤ 32 by default build).
        n_mppi_iters:     MPPI stage iterations.
        n_lbfgs_iters:    L-BFGS stage iterations.
        m_lbfgs:          L-BFGS history size (≤ 8 by default build).
        sigma:            MPPI noise standard deviation [rad/m].
        mppi_temperature: MPPI softmax temperature.
        pos_weight:       Weight on position residual components.
        ori_weight:       Weight on orientation residual components.
        eps_pos:          Position convergence threshold [m].
        eps_ori:          Orientation convergence threshold [rad].
        rng_seed:         Per-launch RNG seed scalar int32 array (mixed with thread/problem index).

    Returns:
        Tuple ``(cfgs, errors)`` where ``cfgs`` has shape
        ``(n_problems, n_seeds, n_act)`` and ``errors`` has shape
        ``(n_problems, n_seeds)``.
    """
    _load_and_register()

    n_problems, n_seeds, n_act = seeds.shape
    seeds = seeds.astype(jnp.float32)
    rb    = _robot_buffers(twists, parent_tf, parent_idx, act_idx,
                           mimic_mul, mimic_off, mimic_act_idx, topo_inv)

    cfgs, errs = jax.ffi.ffi_call(
        "mppi_ik_cuda",
        (
            jax.ShapeDtypeStruct((n_problems, n_seeds, n_act), jnp.float32),
            jax.ShapeDtypeStruct((n_problems, n_seeds),        jnp.float32),
        ),
    )(
        seeds,
        *rb,
        target_jnts.astype(jnp.int32),
        ancestor_masks.astype(jnp.int32),
        target_T.astype(jnp.float32),
        robot_spheres_local.astype(jnp.float32),
        robot_sphere_joint_idx.astype(jnp.int32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
        lower.astype(jnp.float32),
        upper.astype(jnp.float32),
        fixed_mask.astype(jnp.int32),
        rng_seed.astype(jnp.int32),
        n_particles      = int(n_particles),
        n_mppi_iters     = int(n_mppi_iters),
        n_lbfgs_iters    = int(n_lbfgs_iters),
        m_lbfgs          = int(m_lbfgs),
        sigma            = np.float32(sigma),
        mppi_temperature = np.float32(mppi_temperature),
        pos_weight       = np.float32(pos_weight),
        ori_weight       = np.float32(ori_weight),
        eps_pos          = np.float32(eps_pos),
        eps_ori          = np.float32(eps_ori),
        enable_collision = int(bool(enable_collision)),
        collision_weight = np.float32(collision_weight),
        collision_margin = np.float32(collision_margin),
    )
    return cfgs, errs
