# Collision Checking: CUDA Kernel vs JAX Implementation

This document describes how the CUDA collision backend works, how it differs from the JAX
implementation, and explains the numerical discrepancy note:

> `[CUDA uses full SxS min; JAX uses diagonal — expected diff]`

---

## Overview

Pyronot has two collision backends that share the same public API:

| Backend | Class | Robot geometry | FK engine | Distance engine |
|---|---|---|---|---|
| JAX-Capsule | `RobotCollision` | 1 capsule per link | JAX | JAX vmap |
| JAX-Sphere | `RobotCollisionSpherized` | S spheres per link | JAX | JAX vmap |
| CUDA-Capsule | `CUDARobotCollisionChecker(RobotCollision)` | 1 capsule per link | CUDA | CUDA kernels |
| CUDA-Sphere | `CUDARobotCollisionChecker(RobotCollisionSpherized)` | S spheres per link | CUDA | CUDA kernels |

Both backends produce signed distances (positive = separated, negative = penetration) and
expose the same two entry points:

```python
compute_world_collision_distance(robot, cfg, world_geom)  # → [*batch, N, M]
compute_self_collision_distance(robot, cfg)               # → [*batch, P]
```

The CUDA checker wraps a JAX collision object. Both forward kinematics and distance
computation run on CUDA: FK uses the compiled `_fk_cuda_lib.so` kernel via
`robot.forward_kinematics(use_cuda=True)`, and distances use the collision kernels via the
JAX FFI (XLA custom call) interface.

---

## pRRTC Lineage: Adapted and Dropped

The CUDA collision kernel is inspired by [pRRTC](https://github.com/lyf44/pRRTC), a
GPU-parallelised bidirectional RRT planner. This section documents exactly what was
carried over, what was rewritten, and what was removed to fit the pyronot backend.

### What was adapted

**Parallelism strategy**
pRRTC's core insight — dispatch one thread per (batch element, robot sphere, world
obstacle) triplet — is preserved in the world-collision kernels. The 3D grid layout
`(batch, robot_tile, obstacle_tile)` is a direct descendant of pRRTC's kernel
organisation.

**Primitive distance functions**
The sphere–sphere and sphere–capsule distance computations follow the same mathematical
formulation as pRRTC's `math.hh` and `shapes.hh` helpers (originally adapted from
[VAMP](https://github.com/KavrakiLab/vamp)). The segment-closest-point derivation used
for capsule–capsule and sphere–capsule is identical in structure.

**Box/Cuboid parameterisation**
The box is stored as `(center, axis1, axis2, axis3, half_lengths)` — 15 floats — matching
pRRTC's `Cuboid` struct layout. The `box_sdf_local` function computes the same
closest-surface distance.

**SoA memory layout**
pRRTC stores sphere positions as `[sphere_idx, batch, coord]` for coalesced global memory
loads. Pyronot follows the same principle: sphere centers are `[3, B, K]` and capsule data
is `[7, B, N]`, keeping the component axis outermost so consecutive threads read
consecutive memory.

**Type-split world dispatch**
pRRTC routes all world geometry through a single `sphere_environment_in_collision`
function that branches on obstacle type. Pyronot retains the type-split idea but makes it
explicit: a separate kernel is launched per obstacle type (spheres, capsules, boxes,
halfspaces), eliminating per-thread branching entirely.

---

### What was dropped or replaced

**Hardcoded per-robot sphere arrays**
pRRTC hardcodes sphere geometry and joint mappings as `__device__ __constant__` arrays for
each supported robot (Panda, Fetch, Baxter). This makes adding a new robot require
recompiling the kernel. Pyronot replaces this entirely with runtime-provided float32
arrays derived from the URDF via `RobotCollision` / `RobotCollisionSpherized` — no robot
is baked into the kernel source.

**4×4 matrix FK fused into the collision kernel**
pRRTC runs FK inside the collision kernel using a 4×4 rotation matrix accumulated
column-by-column across 4 threads per batch element (one thread per matrix column). FK
and sphere-position update are fused in the same kernel pass. Pyronot separates FK
completely: the standalone `_fk_cuda_kernel.cu` uses SE(3) Lie algebra
(quaternion+translation, `_fk_cuda_helpers.cuh`) and the collision kernels receive
pre-computed world-frame geometry as inputs.

**Fixed block size of 16 configurations**
pRRTC's FK/collision kernels are written around `BATCH_SIZE = 16` (one CUDA block handles
16 robots, 4 threads each). Pyronot uses a fully dynamic batch dimension `B`; block size
is `BLOCK_K = 256` independent of B, and the grid simply tiles over `ceil(B/BLOCK_K)`
blocks.

**Boolean output with early exit**
pRRTC returns a boolean collision result and uses `warp_any` / early-return to abort as
soon as one collision is detected. Pyronot computes a signed floating-point distance for
every (robot geometry, world obstacle) pair with no early exit, because the downstream
optimisers need the full distance field, not just a binary flag.

**Range-based self-collision pairs**
pRRTC encodes self-collision checks as `(sphere_1, range_start, range_end)` triples
hardcoded per robot (e.g. `panda_self_cc_ranges`). Pyronot replaces this with explicit
`(pair_i, pair_j)` index arrays derived from the URDF joint topology and any user-supplied
ignore pairs, supporting arbitrary robots without recompilation.

**C++ `Environment` struct with heap-allocated arrays**
pRRTC passes world geometry through a `ppln::collision::Environment<float>` C++ struct
that owns heap-allocated arrays for each primitive type, requiring `cudaMalloc` /
`cudaMemcpy` / `cudaFree` management. Pyronot replaces this with flat `float32` buffers
passed as typed XLA FFI `Buffer` arguments — lifetime is managed by JAX, and no manual
CUDA memory management is needed in the kernel.

**Axis-aligned shape specialisations**
pRRTC maintains separate `z_aligned_capsules` and `z_aligned_cuboids` slots in the
`Environment` struct for cheaper intersection tests against axis-aligned geometry. Pyronot
drops these: all capsules and boxes go through the general-orientation kernels, which is
sufficient for the obstacle types pyronot currently encounters.

**Cylinder / Capsule distinction**
pRRTC distinguishes `Cylinder` (finite-radius solid) from `Capsule` (hemispherical end
caps) in its type system. Pyronot uses capsules exclusively (two endpoints + radius), and
the collision distance is always the endpoint–endpoint closest-approach formula.

**The planning loop**
pRRTC's primary product is the bidirectional RRTC planner itself — tree management,
Halton sampling, `__device__` volatile state, etc. None of that is ported; only the
collision distance primitives were extracted.

---

### What was added (not in pRRTC)

| Addition | Reason |
|---|---|
| `HalfSpace` world obstacle type | Floor/wall planes common in manipulation tasks |
| Capsule robot geometry | pyronot's capsule-based URDF collision model |
| Fused S-reduction kernel (`wcsr_*`) | Avoids Python-side reshape+min for sphere robots |
| XLA FFI / JAX integration | Required for interoperability with JAX-based planners |
| Signed float distances | Optimisers need gradient-like distance information |
| Arbitrary pair-based self-collision | URDF-derived pair lists, not hardcoded ranges |

---

## JAX Implementation

### World collision

For **capsule-robot** (`RobotCollision`), `compute_world_collision_distance` runs a
`jax.vmap` over the N robot links, calling `collide(link_capsule, world_geom)` for each
one. This produces an `[N, M]` matrix of signed distances per configuration.

For **sphere-robot** (`RobotCollisionSpherized`), each link has S spheres. The method
vectorises over both links (N) and their constituent spheres (S), then reduces with
`min` over S:

```python
# per link: collide each of its S spheres against all M world objects → (S, M)
# then:     min over S  → (M,)
# over all N links via vmap → (N, M)
```

### Self-collision

**Capsule-robot**: builds the full N×N pairwise distance matrix with `pairwise_collide`,
then reads off the pre-computed active pairs `(active_idx_i, active_idx_j)`.

**Sphere-robot** (`RobotCollisionSpherized.compute_self_collision_distance`): also calls
`pairwise_collide(coll, coll)` where `coll` has shape `(S, N)`. The output has shape
`(S, N, S, N)`. It then takes `jnp.min(dist_matrix, axis=0)` to reduce the S-dimension
and obtain an `(N, N)` matrix, from which active pairs are extracted.

**Crucially**: `pairwise_collide` on the `(S, N)` geometry computes distances between
all `(S*N)` spheres simultaneously. The `min(axis=0)` reduction collapses the **first
sphere-index axis** (the "row" S), leaving the result at `[N, N]` by taking the minimum
over the *same sphere index s* pairing: sphere `s` of link `i` vs sphere `s` of link `j`.
This is effectively the **diagonal** of the full S×S pairing — it only compares
sphere 0-to-0, 1-to-1, etc., not sphere 0-of-link-i against sphere 1-of-link-j.

---

## CUDA Kernel Implementation

### Architecture

The CUDA backend (in `_collision_cuda_kernel.cu`) uses a 3D grid strategy:

```
World kernels — blockIdx.x = batch index b
               blockIdx.y = tile over robot dimension (K or N)
               blockIdx.z = tile over obstacle dimension (M_type / TILE_M)
Self kernels  — 1D grid, one thread per (batch b, active pair p)
```

Constants:
- `BLOCK_K = 256` threads per block along the robot dimension
- `TILE_M = 16` obstacles loaded into shared memory per tile

The world geometry is split by type at the Python level (`_extract_world_arrays`). A
separate type-homogeneous kernel is launched for each obstacle type (spheres, capsules,
boxes, halfspaces). This avoids per-thread branching on obstacle type, which would hurt
GPU occupancy.

### Shared-memory tiling

Each block loads `TILE_M` world obstacles into shared memory before computing distances:

```c
// one block loads a tile of world spheres
for (int i = threadIdx.x; i < te * 4; i += BLOCK_K)
    sm[i] = wd[ms * 4 + i];
__syncthreads();
// all 256 threads then compute distances against the same tile
```

This means world geometry is read from global memory only once per tile, amortised across
all `BLOCK_K` robot elements in the block.

### World collision — fused S-reduction kernel (sphere-robot)

For the sphere-robot, the non-reduced path outputs `[B, K, M]` where `K = S * N`. This
requires a Python-side reshape and min to get `[B, N, M]`.

The **fused S-reduction kernel** (`wcsr_vs_*`) avoids that: it tiles over **N links**
instead of K spheres, and each thread loops over all S spheres for its assigned link
before writing the minimum:

```c
float min_d[TILE_M];
for (int s = 0; s < S; s++) {
    int bk = b*K + s*N + n;   // sphere s of link n in batch b
    ...
    for (int t = 0; t < te; t++)
        min_d[t] = fminf(min_d[t], sphere_sphere_dist(...));
}
// write [B, N, M] output directly
out[bn*M + mo+ms+t] = min_d[t];
```

This outputs `[B, N, M]` directly with no Python-side reduction.

### Self-collision — full S×S minimum

The self-collision kernel (`self_collision_sphere_kernel`) gets one thread per
`(batch b, active pair p)`. For each pair `(li, lj)`, the thread iterates over
**all combinations** of `si = 0..S-1` and `sj = 0..S-1`:

```c
for (int si = 0; si < S; si++) {
    ...
    for (int sj = 0; sj < S; sj++) {
        ...
        min_d = fminf(min_d, sphere_sphere_dist(...));
    }
}
out_dist[b*P + p] = min_d;
```

This computes the **true minimum distance** over all S² sphere pairs between the two
links. It is the most geometrically accurate answer: the closest approach between any
sphere of link i and any sphere of link j.

---

## The "full SxS min vs diagonal" discrepancy

This is the meaning of the note printed during the numerical agreement check:

```
[CUDA uses full SxS min; JAX uses diagonal — expected diff]
```

### What "diagonal" means

In the JAX sphere self-collision path, `pairwise_collide(coll, coll)` treats the `(S, N)`
geometry as a flat collection of `S*N` primitives. The resulting distance matrix has shape
`(S, N, S, N)` — entry `[si, li, sj, lj]` is the distance between sphere `si` of link
`li` and sphere `sj` of link `lj`.

When `jnp.min(dist_matrix, axis=0)` is applied, it collapses over the **first** `S` axis
(the row-sphere index `si`), producing `(N, S, N)`. Extracting `[li, :, lj]` then takes
the minimum over `si` only for the *same* `sj`. The net effect is:

```
JAX result for pair (li, lj) = min_{s}  dist(sphere_s_of_li, sphere_s_of_lj)
```

Only paired sphere indices are compared (s=0 vs s=0, s=1 vs s=1, …). This is the
**diagonal** of the S×S cross-product.

### What "full SxS" means

The CUDA kernel computes:

```
CUDA result for pair (li, lj) = min_{si, sj}  dist(sphere_si_of_li, sphere_sj_of_lj)
```

Every sphere of link `i` is compared to every sphere of link `j`. This is the full S²
cross-product — a strictly tighter (smaller or equal) lower bound on separation than the
diagonal.

### Why this matters

If two links have more than one sphere each, the CUDA result will in general be
**smaller or equal** to the JAX result. Concretely:

- CUDA may report a **negative distance** (collision) where JAX reports a **positive
  distance** (clearance), because the closest approach might be between sphere 0 of link i
  and sphere 2 of link j — a pair the JAX path never evaluates.
- The difference is not a bug in either implementation. The CUDA kernel is geometrically
  more correct; the JAX path is an approximation.

For a robot where each link has exactly one sphere (S = 1), the two implementations
are numerically identical since the only possible pairing is (s=0, s=0).

### Summary table

| | JAX sphere self-collision | CUDA sphere self-collision |
|---|---|---|
| Pairs evaluated per link pair | S (diagonal: `s_i = s_j`) | S² (all cross-pairs) |
| Geometric accuracy | Approximation | Exact minimum |
| Result vs CUDA | ≥ CUDA result | True minimum |
| Bug? | No — deliberate simplification | No — intended behavior |

---

## Data Layout

### SoA vs AoS

CUDA kernels use **Structure-of-Arrays (SoA)** layout for world-collision inputs to
maximize coalesced memory access (consecutive threads load consecutive elements of the
same field):

```
sphere centers: [3, B, K]  — all x-coords, then all y-coords, then all z-coords
capsule data:   [7, B, N]  — all x1s, then all y1s, ...
```

The JAX implementations use standard **Array-of-Structures (AoS)** layout that is natural
for JAX pytrees.

The `_cuda_collision.py` layer converts between the two representations inside the JIT
boundary:

```python
# AoS [B, K, 3] → SoA [3, B, K]
centers_soa = jnp.transpose(centers_aoi, (2, 0, 1))
```

### Padding spheres

`RobotCollisionSpherized` pads links that have fewer than `max_spheres` spheres using
dummy spheres with `radius = -1e9`. Both backends skip these during distance computation:

- **CUDA**: the kernel checks `if (rad < 0.0f)` and either skips the sphere (in
  self-collision) or writes the sentinel value `1e9` to the output (in world collision).
- **JAX**: the large negative radius propagates through the SDF arithmetic and produces
  a large positive distance, effectively marking the sphere as non-colliding.

---

## Forward Kinematics: CUDA vs JAX

### How the CUDA FK is invoked

`CUDARobotCollisionChecker._at_config_batched` calls
`robot.forward_kinematics(cfg, use_cuda=True)`, which dispatches to the `fk_cuda` kernel
compiled in `_fk_cuda_lib.so`. The kernel requires the same robot parameters used by the
JAX FK path (twists, parent transforms, topology indices, etc.) and produces an identical
`(*batch, N, 7)` `[wxyz, xyz]` output.

### Batch handling

The CUDA FK kernel natively handles any batch shape, so a single kernel launch covers all
B configurations. The geometry transform (applying link poses to capsules/spheres) is then
vmapped over the batch dimension — this step is cheap and required because
`jdc.copy_and_mutate` (used inside `CollGeom.transform`) enforces that pose shapes don't
change, preventing a single batched transform call from changing the geometry's shape from
`(N,)` to `(B, N,)`.

Previously the JAX FK was called *inside* the vmap — once per batch element. Now:

1. **CUDA FK** runs once for all B elements: `(*batch, N, 7)`.
2. **Geometry transform** is vmapped per element `(N, 7)` → cheap, shape-consistent.

### What must be compiled

The CUDA backend requires two compiled shared libraries:

```bash
bash src/pyronot/cuda_kernels/build_fk_cuda.sh         # _fk_cuda_lib.so
bash src/pyronot/cuda_kernels/build_collision_cuda.sh  # _collision_cuda_lib.so
```

---

## JIT and Caching

The `CUDARobotCollisionChecker` maintains two caches:

1. **World geometry cache** (`set_world` / `_ensure_world_cache`): converts world geometry
   to flat numpy arrays and uploads them as JAX device arrays once. Subsequent calls with
   the same `world_geom` object reuse the device arrays with no host→device copy.

2. **JIT cache** (`_ensure_jit`): builds `jax.jit`-wrapped closures over a specific
   `robot` object. The first call for a given robot triggers XLA tracing and compilation of
   the CUDA FFI calls (both FK and collision). Subsequent calls with the same shapes hit
   the compiled cache.

Both caches are keyed on Python object identity (`id()`), so changing the robot or world
object invalidates the relevant cache and triggers recompilation.
