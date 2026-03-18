"""JAX FFI wrapper for the CUDA collision-distance kernels.

The companion shared library ``_collision_cuda_lib.so`` must be compiled from
``_collision_cuda_kernel.cu`` before this module can be imported:

    bash src/pyronot/cuda_kernels/build_collision_cuda.sh

Requires JAX >= 0.4.14 (for jax.ffi).

Parallelism (pRRTC-style):
  Each collision pair within a batch is handled by a dedicated CUDA thread.
  World collision  — thread per (batch, robot_geom_tile, world_obstacle_tile).
  Self-collision   — thread per (batch, active_pair).

World geometry arrays:
  world_spheres    [Ms, 4]   (x, y, z, r)
  world_capsules   [Mc, 7]   (x1, y1, z1, x2, y2, z2, r)
  world_boxes      [Mb, 15]  (cx,cy,cz, ax1..ax3, ay1..ay3, az1..az3, hl1,hl2,hl3)
  world_halfspaces [Mh, 6]   (nx, ny, nz, px, py, pz)
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_LIB_NAME = "_collision_cuda_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the shared library and register all five FFI targets (runs once)."""
    lib_path = Path(__file__).parent / _LIB_NAME
    if not lib_path.exists():
        raise RuntimeError(
            f"CUDA collision library not found at {lib_path}.\n"
            "Compile it first with:\n"
            "  bash src/pyronot/cuda_kernels/build_collision_cuda.sh\n"
            "(This produces _collision_cuda_lib.so alongside the kernel source.)"
        )
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    def _register(ffi_name: str, symbol_name: str) -> None:
        capsule = _PyCapsule_New(
            ctypes.cast(getattr(lib, symbol_name), ctypes.c_void_p),
            b"xla._CUSTOM_CALL_TARGET",
            None,
        )
        jax.ffi.register_ffi_target(ffi_name, capsule, platform="CUDA")

    _register("collision_world_sphere",         "CollisionWorldSphereFfi")
    _register("collision_world_sphere_reduced", "CollisionWorldSphereReducedFfi")
    _register("collision_world_capsule",        "CollisionWorldCapsuleFfi")
    _register("collision_self_sphere",          "CollisionSelfSphereFfi")
    _register("collision_self_capsule",         "CollisionSelfCapsuleFfi")


def collision_world_sphere(
    sphere_centers:   Array,  # [3, B, K]   float32  SoA layout
    sphere_radii:     Array,  # [B, K]      float32
    world_spheres:    Array,  # [Ms, 4]     float32
    world_capsules:   Array,  # [Mc, 7]     float32
    world_boxes:      Array,  # [Mb, 15]    float32
    world_halfspaces: Array,  # [Mh, 6]     float32
) -> Array:                   # [B, K, M]   float32   M = Ms+Mc+Mb+Mh
    """Signed distances from each robot sphere to every world obstacle.

    sphere_centers is in SoA layout [3, B, K] (component-major).

    One CUDA thread per (batch b, robot sphere k, world obstacle m).
    World obstacles split into type-homogeneous kernels (no per-thread branching).
    """
    _load_and_register()
    B, K = sphere_centers.shape[1], sphere_centers.shape[2]
    Ms = world_spheres.shape[0]
    Mc = world_capsules.shape[0]
    Mb = world_boxes.shape[0]
    Mh = world_halfspaces.shape[0]
    M  = Ms + Mc + Mb + Mh

    if M == 0:
        return jnp.empty((B, K, 0), dtype=jnp.float32)

    return jax.ffi.ffi_call(
        "collision_world_sphere",
        jax.ShapeDtypeStruct((B, K, M), jnp.float32),
    )(
        sphere_centers.astype(jnp.float32),
        sphere_radii.astype(jnp.float32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
    )


def collision_world_sphere_reduced(
    sphere_centers:   Array,  # [3, B, K]   float32  SoA layout, K = S * N
    sphere_radii:     Array,  # [B, K]      float32
    world_spheres:    Array,  # [Ms, 4]     float32
    world_capsules:   Array,  # [Mc, 7]     float32
    world_boxes:      Array,  # [Mb, 15]    float32
    world_halfspaces: Array,  # [Mh, 6]     float32
    n: int,                   # N = number of robot links
) -> Array:                   # [B, N, M]   float32
    """Signed distances from each robot link to every world obstacle, fused S-reduction.

    Unlike ``collision_world_sphere`` (which outputs [B, K, M] requiring a
    Python-side reshape+min to get [B, N, M]), this kernel fuses the per-link
    minimum over S spheres directly in CUDA.  For each (b, n, m), it iterates
    over s=0..S-1, skips padding spheres (radius < 0), and writes the minimum
    distance.

    Grid: (B, ceil(N/BLOCK_K), ceil(M_type/TILE_M)) per obstacle type.
    """
    _load_and_register()
    B, K = sphere_centers.shape[1], sphere_centers.shape[2]
    N = n
    Ms = world_spheres.shape[0]
    Mc = world_capsules.shape[0]
    Mb = world_boxes.shape[0]
    Mh = world_halfspaces.shape[0]
    M  = Ms + Mc + Mb + Mh

    if M == 0:
        return jnp.empty((B, N, 0), dtype=jnp.float32)

    return jax.ffi.ffi_call(
        "collision_world_sphere_reduced",
        jax.ShapeDtypeStruct((B, N, M), jnp.float32),
    )(
        sphere_centers.astype(jnp.float32),
        sphere_radii.astype(jnp.float32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
        n=np.int64(N),
    )


def collision_world_capsule(
    caps:             Array,  # [7, B, N]   float32  SoA layout
    world_spheres:    Array,  # [Ms, 4]     float32
    world_capsules:   Array,  # [Mc, 7]     float32
    world_boxes:      Array,  # [Mb, 15]    float32
    world_halfspaces: Array,  # [Mh, 6]     float32
) -> Array:                   # [B, N, M]   float32   M = Ms+Mc+Mb+Mh
    """Signed distances from each robot capsule link to every world obstacle.

    caps is in SoA layout [7, B, N] (component-major).

    One CUDA thread per (batch b, robot link n, world obstacle m).
    World obstacles split into type-homogeneous kernels.
    """
    _load_and_register()
    B, N = caps.shape[1], caps.shape[2]
    Ms = world_spheres.shape[0]
    Mc = world_capsules.shape[0]
    Mb = world_boxes.shape[0]
    Mh = world_halfspaces.shape[0]
    M  = Ms + Mc + Mb + Mh

    if M == 0:
        return jnp.empty((B, N, 0), dtype=jnp.float32)

    return jax.ffi.ffi_call(
        "collision_world_capsule",
        jax.ShapeDtypeStruct((B, N, M), jnp.float32),
    )(
        caps.astype(jnp.float32),
        world_spheres.astype(jnp.float32),
        world_capsules.astype(jnp.float32),
        world_boxes.astype(jnp.float32),
        world_halfspaces.astype(jnp.float32),
    )


def collision_self_sphere(
    sphere_centers: Array,  # [B, S, N, 3]  float32
    sphere_radii:   Array,  # [B, S, N]     float32
    pair_i:         Array,  # [P]           int32
    pair_j:         Array,  # [P]           int32
) -> Array:                 # [B, P]        float32
    """Minimum signed sphere–sphere distance for each active self-collision pair.

    One CUDA thread per (batch b, active pair p).
    """
    _load_and_register()
    B = sphere_centers.shape[0]
    P = pair_i.shape[0]

    if P == 0:
        return jnp.empty((B, 0), dtype=jnp.float32)

    return jax.ffi.ffi_call(
        "collision_self_sphere",
        jax.ShapeDtypeStruct((B, P), jnp.float32),
    )(
        sphere_centers.astype(jnp.float32),
        sphere_radii.astype(jnp.float32),
        pair_i.astype(jnp.int32),
        pair_j.astype(jnp.int32),
    )


def collision_self_capsule(
    caps:   Array,  # [B, N, 7]  float32
    pair_i: Array,  # [P]        int32
    pair_j: Array,  # [P]        int32
) -> Array:         # [B, P]     float32
    """Signed capsule–capsule distance for each active self-collision pair.

    One CUDA thread per (batch b, active pair p).
    """
    _load_and_register()
    B = caps.shape[0]
    P = pair_i.shape[0]

    if P == 0:
        return jnp.empty((B, 0), dtype=jnp.float32)

    return jax.ffi.ffi_call(
        "collision_self_capsule",
        jax.ShapeDtypeStruct((B, P), jnp.float32),
    )(
        caps.astype(jnp.float32),
        pair_i.astype(jnp.int32),
        pair_j.astype(jnp.int32),
    )


# ── Partial JIT factories ──────────────────────────────────────────────────────
#
# Each factory captures static geometry (world obstacles or collision-pair lists)
# in a closure and returns a jax.jit-compiled callable whose only arguments are
# the *dynamic* robot geometry that changes every call.
#
# JAX compiles once per unique combination of closed-over array shapes and
# reuses the XLA artifact on subsequent calls with the same shapes, so the
# world geometry is effectively a compile-time constant from XLA's perspective.
#
# Usage:
#   check = make_world_sphere_fn(world_spheres, world_capsules, world_boxes, world_hs)
#   dists = check(sphere_centers, sphere_radii)   # JIT-compiled, no world args


def make_world_sphere_fn(
    world_spheres:    Array,  # [Ms, 4]
    world_capsules:   Array,  # [Mc, 7]
    world_boxes:      Array,  # [Mb, 15]
    world_halfspaces: Array,  # [Mh, 6]
) -> Callable[[Array, Array], Array]:
    """Return a JIT-compiled world-sphere checker with pre-bound world geometry.

    The returned function has signature:
        fn(sphere_centers: [3, B, K], sphere_radii: [B, K]) -> [B, K, M]
    """
    @jax.jit
    def _fn(sphere_centers: Array, sphere_radii: Array) -> Array:
        return collision_world_sphere(
            sphere_centers, sphere_radii,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
        )
    return _fn


def make_world_sphere_reduced_fn(
    world_spheres:    Array,  # [Ms, 4]
    world_capsules:   Array,  # [Mc, 7]
    world_boxes:      Array,  # [Mb, 15]
    world_halfspaces: Array,  # [Mh, 6]
    n: int,
) -> Callable[[Array, Array], Array]:
    """Return a JIT-compiled fused-reduction world-sphere checker with pre-bound world/N.

    The returned function has signature:
        fn(sphere_centers: [3, B, K], sphere_radii: [B, K]) -> [B, N, M]
    """
    # n is static (output shape depends on it), so specialise once at factory time.
    @functools.partial(jax.jit, static_argnames=("_n",))
    def _fn(sphere_centers: Array, sphere_radii: Array, *, _n: int) -> Array:
        return collision_world_sphere_reduced(
            sphere_centers, sphere_radii,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n=_n,
        )
    return functools.partial(_fn, _n=n)


def make_world_capsule_fn(
    world_spheres:    Array,  # [Ms, 4]
    world_capsules:   Array,  # [Mc, 7]
    world_boxes:      Array,  # [Mb, 15]
    world_halfspaces: Array,  # [Mh, 6]
) -> Callable[[Array], Array]:
    """Return a JIT-compiled world-capsule checker with pre-bound world geometry.

    The returned function has signature:
        fn(caps: [7, B, N]) -> [B, N, M]
    """
    @jax.jit
    def _fn(caps: Array) -> Array:
        return collision_world_capsule(
            caps,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
        )
    return _fn


def make_self_sphere_fn(
    pair_i: Array,  # [P]  int32
    pair_j: Array,  # [P]  int32
) -> Callable[[Array, Array], Array]:
    """Return a JIT-compiled self-sphere checker with pre-bound collision pairs.

    The returned function has signature:
        fn(sphere_centers: [B, S, N, 3], sphere_radii: [B, S, N]) -> [B, P]
    """
    @jax.jit
    def _fn(sphere_centers: Array, sphere_radii: Array) -> Array:
        return collision_self_sphere(sphere_centers, sphere_radii, pair_i, pair_j)
    return _fn


def make_self_capsule_fn(
    pair_i: Array,  # [P]  int32
    pair_j: Array,  # [P]  int32
) -> Callable[[Array], Array]:
    """Return a JIT-compiled self-capsule checker with pre-bound collision pairs.

    The returned function has signature:
        fn(caps: [B, N, 7]) -> [B, P]
    """
    @jax.jit
    def _fn(caps: Array) -> Array:
        return collision_self_capsule(caps, pair_i, pair_j)
    return _fn
