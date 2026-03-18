"""Learned IK Solver: Conditional Normalizing Flow (IKFlow) warm-start
+ Levenberg-Marquardt refinement.

Architecture
------------
A conditional Real-NVP normalizing flow with affine coupling layers (Glow-style)
models the conditional distribution p(q | pose).  The design follows IKFlow
(Ames et al., RA-L 2022):

  * Each coupling layer uses a 3 × 1024 MLP subnet (Leaky-ReLU).
  * The flow operates in a latent space of dimension ``latent_dim`` (≥ n_act).
    Joint configs are zero-padded to ``latent_dim`` before the forward pass and
    the first ``n_act`` elements are taken after the inverse pass.  The extra
    dimensions improve modelling of multimodal IK solution sets.
  * Fixed random permutations between coupling layers (Glow-style) ensure that
    every original dimension gets mixed across layers.  All coupling layers use
    the same split (fix first half, transform second half); mixing is provided
    by the permutations rather than by alternating masks.
  * The affine transformation uses tanh-bounded log-scales:
        y_trans = x_trans * exp(tanh(s)) + t,   (s, t) = subnet([x_fixed, cond])

SoftFlow (training)
-------------------
To prevent training divergence on lower-dimensional IK solution manifolds
(paper Section III-B), Gaussian noise scaled by a random scalar ``c`` is
injected into joint configs during training, and ``c`` is appended to the
conditioning vector.  At inference, ``c = 0`` is appended.

    c ~ U(0, softflow_scale),   v ~ N(0, I),   q̃ = q + c·v
    cond_train = [pose_enc, c],   cond_infer = [pose_enc, 0.0]

Inference
---------
Latent samples z ~ N(0, I) are scaled by ``latent_scale = 0.25`` (paper default)
before the inverse flow to trade diversity for accuracy.  Multiple seeds decoded
from the flow are then refined by Levenberg-Marquardt.

Training
--------
Minimise NLL of the base Gaussian after change-of-variables:
    NLL(q | pose) = 0.5 * ||z||² − log|det J|

Pose encoding
-------------
Each EE pose is encoded as 12-dim [R_flat(9), t(3)].  Multi-EE encodings are
concatenated.  A scalar SoftFlow conditioning value c is appended (total dim
12*n_ee + 1).

Reference
---------
Ames, B., Morgan, J., Konidaris, G. (2022). IKFlow: Generating Diverse Inverse
Kinematics Solutions. IEEE RA-L 7(3), 7177–7184. arXiv:2111.08933.

Requires
--------
    pip install flax
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import flax.linen as nn
except ImportError as err:
    raise ImportError(
        "The learned IK solver requires Flax.  Install it with:\n"
        "    pip install flax"
    ) from err

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _ik_residual
from ._ls_ik import _ls_ik_single


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class _CouplingNet(nn.Module):
    """3-hidden-layer 1024-wide subnet for one affine coupling layer.

    Input:  concatenation of the fixed half of the latent vector and the pose
            conditioning (plus SoftFlow scalar c).
    Output: (log_scale, shift) for the transformed half, shape (2 * n_trans,).

    Args:
        n_out:          Output dimension = 2 * n_transformed.
        hidden:         Width of each hidden layer (default 1024).
        negative_slope: Leaky-ReLU negative slope (default 0.01).
    """

    n_out: int
    hidden: int = 1024
    negative_slope: float = 0.01

    @nn.compact
    def __call__(self, x: Float[Array, "n_in"]) -> Float[Array, "n_out"]:
        for _ in range(3):
            x = nn.Dense(self.hidden)(x)
            x = nn.leaky_relu(x, negative_slope=self.negative_slope)
        return nn.Dense(self.n_out)(x)


class IKFlowNet(nn.Module):
    """Conditional Glow-style normalizing flow for IK (IKFlow architecture).

    The flow operates in a ``latent_dim``-dimensional space (≥ n_act).  Joint
    configs are zero-padded to ``latent_dim`` before the forward pass; the
    inverse pass returns the first ``n_act`` elements.

    All coupling layers fix the first ``latent_dim // 2`` dimensions and
    transform the remaining ones.  Fixed random permutations between layers
    (seeded deterministically at 0) provide the dimension mixing.

    Args:
        n_act:      Number of actuated joints.
        latent_dim: Latent space dimension (≥ n_act; paper uses 15 for 7-8 DOF).
        n_layers:   Number of affine coupling layers (default 15).
        hidden:     Hidden-layer width of each subnet (default 1024).
    """

    n_act: int
    latent_dim: int = 15
    n_layers: int = 15
    hidden: int = 1024

    def setup(self) -> None:
        assert self.latent_dim >= self.n_act, (
            f"latent_dim ({self.latent_dim}) must be >= n_act ({self.n_act})"
        )
        d1 = self.latent_dim // 2
        d2 = self.latent_dim - d1
        # All coupling nets: same fixed split (fix d1, transform d2).
        # Glow-style random permutations between layers provide the mixing.
        self.nets = [
            _CouplingNet(n_out=2 * d2, hidden=self.hidden)
            for _ in range(self.n_layers)
        ]
        # Deterministic fixed permutations (seed=0); stored as Python tuples
        # so JAX treats the indices as compile-time constants.
        _rng = np.random.default_rng(0)
        _perms = [_rng.permutation(self.latent_dim) for _ in range(self.n_layers - 1)]
        self._perms     = [tuple(int(x) for x in p)          for p in _perms]
        self._inv_perms = [tuple(int(x) for x in np.argsort(p)) for p in _perms]

    # ------------------------------------------------------------------
    # Internal helpers (single-sample, no batch dim)
    # ------------------------------------------------------------------

    def _layer_forward(
        self, i: int, x: Float[Array, "latent_dim"], cond: Float[Array, "n_cond"]
    ) -> tuple[Float[Array, "latent_dim"], Array]:
        d1 = self.latent_dim // 2
        d2 = self.latent_dim - d1
        x_fixed, x_trans = x[:d1], x[d1:]
        st = self.nets[i](jnp.concatenate([x_fixed, cond]))  # (2*d2,)
        s, t = st[:d2], st[d2:]
        s = jnp.tanh(s)                                       # bounded log-scale
        return jnp.concatenate([x_fixed, x_trans * jnp.exp(s) + t]), jnp.sum(s)

    def _layer_inverse(
        self, i: int, y: Float[Array, "latent_dim"], cond: Float[Array, "n_cond"]
    ) -> Float[Array, "latent_dim"]:
        d1 = self.latent_dim // 2
        d2 = self.latent_dim - d1
        y_fixed, y_trans = y[:d1], y[d1:]
        st = self.nets[i](jnp.concatenate([y_fixed, cond]))
        s, t = st[:d2], st[d2:]
        s = jnp.tanh(s)
        return jnp.concatenate([y_fixed, (y_trans - t) * jnp.exp(-s)])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        q_norm: Float[Array, "n_act"],
        cond:   Float[Array, "n_cond"],
    ) -> tuple[Float[Array, "latent_dim"], Array]:
        """Forward pass: q_norm → z.

        Zero-pads ``q_norm`` to ``latent_dim``, applies coupling layers with
        Glow-style permutations between them, returns the latent code ``z``
        and the total log |det J| (for NLL training).

        Args:
            q_norm: Normalised joint configuration in (−1, 1)^n_act.
            cond:   Conditioning vector: [pose_enc (12*n_ee), c_softflow (1)].

        Returns:
            z:             Latent code of shape (latent_dim,).
            total_log_det: Scalar log |det J| of the full transformation.
        """
        n_pad = self.latent_dim - self.n_act
        x = jnp.concatenate([q_norm, jnp.zeros(n_pad, dtype=q_norm.dtype)])
        total_log_det = jnp.zeros(())
        for i in range(self.n_layers):
            x, ld = self._layer_forward(i, x, cond)
            total_log_det = total_log_det + ld
            if i < self.n_layers - 1:
                x = x[jnp.array(self._perms[i])]
        return x, total_log_det

    def inverse(
        self,
        z:    Float[Array, "latent_dim"],
        cond: Float[Array, "n_cond"],
    ) -> Float[Array, "n_act"]:
        """Inverse pass: z → q_norm.

        Applies coupling layers in reverse order (with inverse permutations),
        then returns the first ``n_act`` elements as the normalised joint config.

        Args:
            z:    Latent code of shape (latent_dim,); typically scaled N(0, I).
            cond: Conditioning vector: [pose_enc (12*n_ee), 0.0] at inference.

        Returns:
            q_norm: Normalised joint configuration in (approximately) (−1, 1)^n_act.
        """
        x = z
        for i in reversed(range(self.n_layers)):
            x = self._layer_inverse(i, x, cond)
            if i > 0:
                x = x[jnp.array(self._inv_perms[i - 1])]
        return x[:self.n_act]


# ---------------------------------------------------------------------------
# Pose encoding helpers
# ---------------------------------------------------------------------------

def encode_pose(pose: jaxlie.SE3) -> Float[Array, "12"]:
    """Encode a single SE(3) pose as a 12-dimensional float vector.

    Layout: [R_rowmajor(9), t(3)].
    """
    R = pose.rotation().as_matrix().reshape(-1)  # (9,)  elements in [-1, 1]
    t = pose.translation()                        # (3,)  metres
    return jnp.concatenate([R, t])                # (12,)


def encode_poses(target_poses: tuple) -> Float[Array, "12_n_ee"]:
    """Encode a tuple of SE(3) poses into a concatenated feature vector.

    Single EE → 12 dims; N EEs → 12*N dims.
    """
    return jnp.concatenate([encode_pose(p) for p in target_poses])


# ---------------------------------------------------------------------------
# Factory: returns a JIT-compiled solver bound to a specific robot
# ---------------------------------------------------------------------------

def make_learned_ik_solve(
    robot: Robot,
    latent_dim: int = 15,
    n_layers: int = 15,
    hidden: int = 1024,
):
    """Create a JIT-compiled learned-IK solver bound to *robot*.

    Args:
        robot:      The robot model.
        latent_dim: Latent space dimension used when the model was trained
                    (default 15, matching the IKFlow paper).  Must match the
                    ``latent_dim`` stored in the model checkpoint.
        n_layers:   Number of affine coupling layers in the trained model.
        hidden:     Width of the hidden layers used during training.

    Returns:
        A JIT-compiled callable::

            learned_ik_solve(
                robot, target_link_indices, target_poses,
                rng_key, previous_cfg, model_params,
                *, num_seeds, n_refine_iters, pos_weight, ori_weight,
                lambda_init, continuity_weight, fixed_joint_mask,
                latent_scale,
            ) -> cfg  # shape (n_act,)
    """
    n_act = int(robot.joints.num_actuated_joints)
    net = IKFlowNet(n_act=n_act, latent_dim=latent_dim, n_layers=n_layers, hidden=hidden)

    @functools.partial(
        jax.jit,
        static_argnames=("target_link_indices", "num_seeds", "n_refine_iters"),
    )
    def learned_ik_solve(
        robot:               Robot,
        target_link_indices: tuple[int, ...],
        target_poses:        tuple,
        rng_key:             Array,
        previous_cfg:        Float[Array, "n_act"],
        model_params:        Any,
        num_seeds:           int   = 16,
        n_refine_iters:      int   = 15,
        pos_weight:          float = 50.0,
        ori_weight:          float = 10.0,
        lambda_init:         float = 5e-3,
        continuity_weight:   float = 0.0,
        fixed_joint_mask:    Float[Array, "n_act"] | None = None,
        latent_scale:        float = 0.25,
    ) -> Float[Array, "n_act"]:
        """Solve IK: flow warm-start → multi-seed LM refinement → winner.

        Stage 1 — Flow sampling
            Encode the target pose(s), append SoftFlow scalar c=0 (inference
            convention), and draw ``num_seeds // 2`` latent codes
            z ~ N(0, I) * latent_scale.  Each is decoded via the inverse flow
            to produce diverse IK candidates from the learned distribution.
            The remaining ``num_seeds // 2`` seeds are uniform random for
            coverage.

        Stage 2 — Multi-seed LM refinement
            All seeds are refined in parallel via ``jax.vmap``.

        Stage 3 — Winner selection
            Seed with lowest weighted SE(3) residual (+ continuity penalty) wins.

        Args:
            latent_scale: Scale applied to latent samples before inverse flow
                          (paper default 0.25; lower → more accurate, less diverse).
        """
        lower = robot.joints.lower_limits
        upper = robot.joints.upper_limits
        mid   = (lower + upper) * 0.5
        half  = (upper - lower) * 0.5

        if fixed_joint_mask is None:
            fixed_joint_mask = jnp.zeros(n_act, dtype=jnp.bool_)

        # Append c=0 for SoftFlow at inference (models were trained with c appended)
        pose_enc_raw = encode_poses(target_poses).astype(jnp.float32)
        pose_enc = jnp.append(pose_enc_raw, jnp.zeros(1, dtype=jnp.float32))

        # ── Stage 1: Flow sampling ────────────────────────────────────────
        n_flow = max(1, num_seeds // 2)
        n_rand = num_seeds - n_flow
        key_f, key_r = jax.random.split(rng_key)

        # Scale truncates the tails of the base distribution (paper: 0.25)
        zs = jax.random.normal(key_f, (n_flow, latent_dim), dtype=jnp.float32) * latent_scale
        q_norms = jax.vmap(
            lambda z: net.apply(model_params, z, pose_enc, method=IKFlowNet.inverse)
        )(zs)                                                # (n_flow, n_act)
        flow_seeds = jnp.clip(mid + half * q_norms, lower, upper)
        flow_seeds = jnp.where(fixed_joint_mask[None], previous_cfg[None], flow_seeds)

        rand_seeds = jax.random.uniform(key_r, (n_rand, n_act), minval=lower, maxval=upper)
        rand_seeds = jnp.where(fixed_joint_mask[None], previous_cfg[None], rand_seeds)

        seeds = jnp.concatenate([flow_seeds, rand_seeds], axis=0)  # (num_seeds, n_act)

        # ── Stage 2: LM refinement ────────────────────────────────────────
        all_cfgs = jax.vmap(
            lambda cfg: _ls_ik_single(
                cfg, robot, target_link_indices, target_poses,
                n_refine_iters, lambda_init, pos_weight, ori_weight,
                lower, upper, fixed_joint_mask,
            )
        )(seeds)

        # ── Stage 3: Winner selection ─────────────────────────────────────
        W = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])

        def weighted_err(cfg: Array) -> Array:
            task_err = sum(
                jnp.sum(
                    (_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) * W) ** 2
                )
                for i in range(len(target_link_indices))
            )
            return task_err + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)

        errors   = jax.vmap(weighted_err)(all_cfgs)
        best_idx = jnp.argmin(errors)
        return all_cfgs[best_idx]

    return learned_ik_solve


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_learned_ik(
    path: str | Path,
    params: Any,
    robot_name: str,
    target_link_names: list[str],
    target_link_indices: list[int],
    n_act: int,
    latent_dim: int = 15,
) -> None:
    """Pickle the Flax params together with robot metadata.

    Args:
        path:                File to write (will be created/overwritten).
        params:              Flax parameter dict (``IKFlowNet.init`` output).
        robot_name:          Name of the robot (e.g. ``"panda"``).
        target_link_names:   List of target link names (one per EE).
        target_link_indices: List of target link indices (one per EE).
        n_act:               Number of actuated joints.
        latent_dim:          Latent space dimension used during training.
    """
    def _to_numpy(x):
        if hasattr(x, "__jax_array__"):
            return np.array(x)
        if isinstance(x, dict):
            return {k: _to_numpy(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_to_numpy(v) for v in x)
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "params":              _to_numpy(params),
        "robot_name":          robot_name,
        "target_link_names":   list(target_link_names),
        "target_link_indices": list(target_link_indices),
        "n_act":               n_act,
        "n_ee":                len(target_link_indices),
        "latent_dim":          latent_dim,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=5)


def load_learned_ik(path: str | Path) -> dict:
    """Load a saved learned-IK model file.

    Returns a dict with keys: ``"params"``, ``"robot_name"``,
    ``"target_link_names"``, ``"target_link_indices"``, ``"n_act"``,
    ``"n_ee"``, ``"latent_dim"``.

    Pass ``data["params"]`` as ``model_params`` and ``data["latent_dim"]``
    to ``make_learned_ik_solve`` to reconstruct the solver.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def get_default_model_path(robot_name: str) -> Path:
    """Return the canonical save path for a robot's learned-IK model."""
    resources = (
        Path(__file__).parent  # optimization_engines/
        .parent                # pyronot/
        .parent                # src/
        .parent                # project root
        / "resources"
        / "learned_ik"
    )
    return resources / f"{robot_name}.pkl"
