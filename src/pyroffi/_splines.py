"""Spline utilities for trajectory initialisation and parameterisation.

Three interpolation modes are provided, all returning dense waypoint arrays
of shape ``[T, DOF]`` that slot directly into ``sco_trajopt`` as ``init_trajs``
rows or as trajectory representations during optimisation.

Interpolation modes
-------------------
``linear_interpolate``
    Piecewise-linear interpolation through an ordered sequence of control
    points.  Zero-order smooth; cheap and always produces a valid path.

``cubic_spline_interpolate``
    Natural cubic spline through the control points.  C² continuous — ideal
    for smooth initialisation.  Tridiagonal system solved in closed form via
    ``jnp.linalg.solve`` so the whole function is JIT-compilable.

``bspline_interpolate``
    Uniform B-spline with clamped end conditions that passes through (or near)
    the provided control points.  The degree is configurable (default 3 /
    cubic).  Produces C^{degree-1} curves; useful when you want a smooth
    *parameterisation* rather than exact interpolation.

Batched helpers
---------------
``make_spline_init_trajs``
    Creates a ``[B, T, DOF]`` batch for ``sco_trajopt`` by generating B copies
    of the chosen spline and adding independent Gaussian noise to each.

All functions are pure JAX and fully JIT-compilable when called with static
``n_points`` / ``degree`` arguments.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------

def linear_interpolate(
    control_points: Float[Array, "K DOF"],
    n_points: int,
) -> Float[Array, "T DOF"]:
    """Piecewise-linear interpolation through ``control_points``.

    Args:
        control_points: Ordered waypoints to interpolate through.  Shape ``[K, DOF]``.
        n_points:       Number of output samples ``T`` (including endpoints).

    Returns:
        Dense trajectory of shape ``[T, DOF]``.
    """
    k = control_points.shape[0]
    # Parameter t in [0, K-1] for each output point
    t_out = jnp.linspace(0.0, k - 1, n_points)   # [T]

    # Segment index and local fraction
    seg   = jnp.clip(jnp.floor(t_out).astype(jnp.int32), 0, k - 2)  # [T]
    alpha = t_out - seg.astype(jnp.float32)                           # [T]

    p0 = control_points[seg]       # [T, DOF]
    p1 = control_points[seg + 1]   # [T, DOF]
    return p0 + alpha[:, None] * (p1 - p0)


# ---------------------------------------------------------------------------
# Cubic spline (natural boundary conditions)
# ---------------------------------------------------------------------------

def _build_natural_cubic_spline_coeffs(
    control_points: Float[Array, "K DOF"],
) -> tuple[
    Float[Array, "K-1 DOF"],
    Float[Array, "K-1 DOF"],
    Float[Array, "K-1 DOF"],
    Float[Array, "K-1 DOF"],
]:
    """Compute natural cubic spline coefficients.

    For K control points we have K-1 segments.  Each segment ``i`` is
    parameterised in ``s ∈ [0, 1]`` as::

        p(s) = a[i] + b[i]*s + c[i]*s² + d[i]*s³

    Natural boundary conditions: second derivative = 0 at both ends.

    Returns:
        (a, b, c, d) each of shape ``[K-1, DOF]``.
    """
    k   = control_points.shape[0]
    n   = k - 1   # number of segments

    # --- Build tridiagonal system for the second derivatives (moments) M ---
    # h[i] = 1 for uniformly spaced knots
    # System:  [2 1        ] [M_1  ]   [6*(y_{i+1}-2y_i+y_{i-1})]
    #          [1 2 1      ] [M_2  ] = [                         ]
    #          [  ...      ] [... ]
    # with M_0 = M_{n} = 0 (natural BCs)
    # We solve for M_1 … M_{n-1}.

    rhs_full = 6.0 * (
        control_points[2:] - 2.0 * control_points[1:-1] + control_points[:-2]
    )  # [K-2, DOF]

    # Tridiagonal matrix: symmetric, diag=4 (for interior), off-diag=1.
    # (Using h_i = 1 everywhere: 2*(h_{i-1}+h_i) = 4, and h_{i-1}=h_i=1.)
    diag    = jnp.full(k - 2, 4.0)
    offdiag = jnp.ones(k - 3)

    A = jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)  # [K-2, K-2]

    # Solve independently for each DOF dimension.
    # M_interior shape: [K-2, DOF]
    M_interior = jnp.linalg.solve(A, rhs_full)   # [K-2, DOF]

    # Pad with zeros for the boundary moments
    zeros = jnp.zeros((1, control_points.shape[1]))
    M = jnp.concatenate([zeros, M_interior, zeros], axis=0)  # [K, DOF]

    # Compute coefficients for each segment
    y  = control_points        # [K, DOF]
    a  = y[:-1]                # [K-1, DOF]
    b  = (y[1:] - y[:-1]) - (2.0 * M[:-1] + M[1:]) / 6.0   # [K-1, DOF]
    c  = M[:-1] / 2.0                                         # [K-1, DOF]
    d  = (M[1:] - M[:-1]) / 6.0                              # [K-1, DOF]

    return a, b, c, d


def cubic_spline_interpolate(
    control_points: Float[Array, "K DOF"],
    n_points: int,
) -> Float[Array, "T DOF"]:
    """Natural cubic spline interpolation through ``control_points``.

    The spline is C² everywhere and satisfies zero second-derivative boundary
    conditions at both ends.

    Args:
        control_points: Ordered waypoints.  Shape ``[K, DOF]``.  K ≥ 2.
            Falls back to linear interpolation when K = 2.
        n_points:       Number of output samples ``T``.

    Returns:
        Dense trajectory of shape ``[T, DOF]``.
    """
    k = control_points.shape[0]

    # K=2: cubic spline degenerates to linear — avoid empty tridiagonal system.
    if k < 3:
        return linear_interpolate(control_points, n_points)

    a, b, c, d = _build_natural_cubic_spline_coeffs(control_points)

    # Query parameter in [0, K-1]
    t_out = jnp.linspace(0.0, k - 1, n_points)          # [T]
    seg   = jnp.clip(jnp.floor(t_out).astype(jnp.int32), 0, k - 2)  # [T]
    s     = t_out - seg.astype(t_out.dtype)               # local param in [0,1]

    # Evaluate cubic: a + b*s + c*s² + d*s³
    result = (
        a[seg]
        + b[seg] * s[:, None]
        + c[seg] * (s ** 2)[:, None]
        + d[seg] * (s ** 3)[:, None]
    )  # [T, DOF]
    return result


# ---------------------------------------------------------------------------
# Uniform clamped B-spline
# ---------------------------------------------------------------------------

def _uniform_bspline_basis(
    t: Float[Array, " T"],
    n_ctrl: int,
    degree: int,
) -> Float[Array, "T n_ctrl"]:
    """Evaluate the uniform B-spline basis matrix via Cox–de Boor recursion.

    The recursion is unrolled with a Python for-loop (``degree`` is static),
    which avoids the dynamic-slice issue that arises when ``p`` is a traced
    value inside ``lax.scan``.  Each iteration reduces the column count by 1,
    so the array shapes differ across iterations — another reason ``lax.scan``
    is unsuitable here.

    Knot vector: clamped uniform — ``degree`` repeated knots at each end,
    uniform in the interior.

    Args:
        t:       Query parameters in ``[0, 1]``.  Shape ``[T]``.
        n_ctrl:  Number of control points.
        degree:  Polynomial degree (≥ 1).  Must be a Python int (static).

    Returns:
        Basis matrix of shape ``[T, n_ctrl]``.
    """
    n_spans = n_ctrl - degree          # number of interior spans
    n_knots = n_ctrl + degree + 1      # total knots

    # Clamped uniform knot vector: degree zeros, uniform interior, degree ones.
    interior = jnp.linspace(0.0, 1.0, n_spans + 1)
    knots = jnp.concatenate([
        jnp.zeros(degree),
        interior,
        jnp.ones(degree),
    ])  # [n_knots]  — all sizes are static Python ints

    # Clamp t slightly below 1 so the last span is included.
    t = jnp.clip(t, 0.0, 1.0 - 1e-7)
    t_col = t[:, None]   # [T, 1]

    # --- Degree-0 basis -------------------------------------------------------
    # N[:, j] = 1  iff  knots[j] <= t < knots[j+1]
    # Shape: [T, n_knots-1]
    N = jnp.where(
        (t_col >= knots[None, :-1]) & (t_col < knots[None, 1:]),
        1.0,
        0.0,
    )

    # --- Cox–de Boor recursion (Python loop — p is a static int) -------------
    # After iteration p the shape is [T, n_knots-1-p].
    # Final shape after `degree` iterations: [T, n_knots-1-degree] = [T, n_ctrl].
    for p in range(1, degree + 1):
        # N_{i,p} needs knots[i], knots[i+p], knots[i+p+1], knots[i+1]
        # for i = 0 … n_knots-2-p  →  (n_knots-1-p) basis functions at level p.
        n_basis = N.shape[1] - 1          # number of output basis functions

        left_num  = t_col        - knots[None, :n_basis]           # [T, n_basis]
        left_den  = knots[None, p:p + n_basis] - knots[None, :n_basis]  # [T, n_basis]
        right_num = knots[None, p + 1:p + 1 + n_basis] - t_col    # [T, n_basis]
        right_den = knots[None, p + 1:p + 1 + n_basis] - knots[None, 1:1 + n_basis]  # [T, n_basis]

        safe_ld = jnp.where(left_den  == 0.0, 1.0, left_den)
        safe_rd = jnp.where(right_den == 0.0, 1.0, right_den)

        left_coef  = jnp.where(left_den  == 0.0, 0.0, left_num  / safe_ld)
        right_coef = jnp.where(right_den == 0.0, 0.0, right_num / safe_rd)

        N = left_coef * N[:, :-1] + right_coef * N[:, 1:]
        # N shape is now [T, n_knots-1-p]

    # After `degree` iterations: N shape is [T, n_ctrl]
    return N


def bspline_interpolate(
    control_points: Float[Array, "K DOF"],
    n_points: int,
    degree: int = 3,
) -> Float[Array, "T DOF"]:
    """Uniform clamped B-spline through (approximately) ``control_points``.

    The curve passes exactly through the first and last control point.
    Interior control points act as *attractors* rather than exact interpolants.
    This gives a smooth, globally supported curve suitable for trajectory
    initialisation.

    Args:
        control_points: Control polygon vertices.  Shape ``[K, DOF]``.
            Requires ``K > degree``.
        n_points:       Number of output samples ``T``.
        degree:         Polynomial degree.  Default 3 (cubic).

    Returns:
        Dense trajectory of shape ``[T, DOF]``.
    """
    k = control_points.shape[0]
    assert k > degree, f"Need more control points ({k}) than degree ({degree})."

    t    = jnp.linspace(0.0, 1.0, n_points)   # [T]
    N    = _uniform_bspline_basis(t, k, degree)  # [T, K]
    traj = N @ control_points                   # [T, DOF]
    return traj


# ---------------------------------------------------------------------------
# Batched init helper for sco_trajopt
# ---------------------------------------------------------------------------

SplineMode = Literal["linear", "cubic", "bspline"]


def make_spline_init_trajs(
    control_points: Float[Array, "K DOF"],
    n_batch: int,
    n_points: int,
    key: Array,
    mode: SplineMode = "cubic",
    noise_scale: float = 0.05,
    bspline_degree: int = 3,
) -> Float[Array, "B T DOF"]:
    """Create a ``[B, T, DOF]`` batch of noisy spline trajectories for ``sco_trajopt``.

    Generates a single base spline from ``control_points``, then tiles it B
    times and adds independent Gaussian noise to produce diverse initial seeds.

    Args:
        control_points: Ordered keyframes.  Shape ``[K, DOF]``.
        n_batch:        Number of candidate trajectories ``B``.
        n_points:       Waypoints per trajectory ``T``.
        key:            JAX PRNG key.
        mode:           ``"linear"``, ``"cubic"``, or ``"bspline"``.
        noise_scale:    Standard deviation of the per-trajectory additive noise.
        bspline_degree: Polynomial degree (only used when ``mode="bspline"``).

    Returns:
        Trajectory batch of shape ``[B, T, DOF]``.
    """
    if mode == "linear":
        base = linear_interpolate(control_points, n_points)
    elif mode == "cubic":
        base = cubic_spline_interpolate(control_points, n_points)
    elif mode == "bspline":
        # Clamp degree so it's always < number of control points.
        bspline_degree = min(bspline_degree, control_points.shape[0] - 1)
        base = bspline_interpolate(control_points, n_points, degree=bspline_degree)
    else:
        raise ValueError(f"Unknown spline mode: {mode!r}. Choose 'linear', 'cubic', or 'bspline'.")

    # Tile and perturb: [B, T, DOF]
    trajs = jnp.broadcast_to(base[None], (n_batch, n_points, control_points.shape[1]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise
