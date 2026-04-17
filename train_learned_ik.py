"""Train a Learned IK model for a given robot.

Generates random joint configurations, computes forward kinematics to obtain
target poses, and trains a conditional normalizing flow (IKFlow architecture)
to model the distribution p(q | pose).  The trained model is saved to disk for
later use by the learned-IK solver in ``_learned_ik.py``.

Training details match the IKFlow paper (Ames et al., RA-L 2022):
  - Optimizer:   RAdam, LR = 5e-4
  - LR schedule: exponential decay, rate 0.979 every 39 000 steps
  - Batch size:  128
  - Dataset:     2.5 M random (q, FK(q)) pairs per robot
  - Loss:        NLL of the base Gaussian after change-of-variables
  - SoftFlow:    per-batch noise injection (scale 1e-3) + c appended to cond

Multi-EE support
----------------
For robots with multiple end-effectors (e.g. Baxter's two hands), all EE
poses are encoded and concatenated into a single conditioning vector.  The
conditioning dimension is 12 × n_ee + 1 (12 per EE × n_ee EEs, plus the
SoftFlow scalar c).  At inference, c = 0 is appended.

Pipeline
--------
1. Load robot via ``robot_descriptions``.
2. Sample ``n_train`` random configurations (uniformly in joint limits).
3. Compute FK in batches; extract poses for all target EEs.
4. Encode each EE pose as 12-dim (rotation matrix + translation) and
   concatenate → (12 × n_ee) conditioning per sample.
5. Normalise joint targets to (−1, 1) for the normalizing flow.
6. Train with RAdam + exponential-decay LR (NLL + SoftFlow loss).
7. Evaluate sampled IK quality (flow sampling only, before LM refinement).
8. Save params + metadata as a pickle file in ``resources/learned_ik/``.

Usage
-----
Train panda (default, single EE):
    python train_learned_ik.py

Train all four benchmark robots:
    python train_learned_ik.py --robot panda
    python train_learned_ik.py --robot baxter        # trains on BOTH hands
    python train_learned_ik.py --robot fetch
    python train_learned_ik.py --robot ur5

Override the target link(s) for an unknown robot:
    python train_learned_ik.py --robot myrobot --target_links tool_link
    python train_learned_ik.py --robot myrobot --target_links left_ee right_ee

List available link names (no training):
    python train_learned_ik.py --robot fetch --list_links

Custom output path:
    python train_learned_ik.py --robot panda --model_dir /tmp/models

Requires
--------
    pip install flax optax
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

try:
    import flax.linen as nn
    import optax
    from flax.training import train_state as flax_train_state
except ImportError as err:
    raise ImportError(
        "Training requires flax and optax.  Install them with:\n"
        "    pip install flax optax"
    ) from err

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroffi as pk
from pyroffi.optimization_engines._learned_ik import (
    IKFlowNet,
    encode_pose,
    get_default_model_path,
    save_learned_ik,
)

# ---------------------------------------------------------------------------
# Per-robot default configurations
# ---------------------------------------------------------------------------

#: Known robot configurations.
#: ``target_links`` is a list so that multi-EE robots (Baxter) can list all
#: end-effectors.  Single-EE robots use a one-element list.
ROBOT_CONFIGS: dict[str, dict] = {
    "panda": {
        "target_links":      ["panda_hand"],
        "fixed_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
    },
    "baxter": {
        # Both hands are modelled jointly: the 24-dim input encodes the
        # right-hand pose followed by the left-hand pose.
        "target_links":      ["right_hand", "left_hand"],
        "fixed_joint_names": [],
    },
    "fetch": {
        "target_links":      ["gripper_link"],
        "fixed_joint_names": [],
    },
    "ur5": {
        "target_links":      ["ee_link"],
        "fixed_joint_names": [],
    },
}


# ---------------------------------------------------------------------------
# Training configuration dataclass (parsed by tyro)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    """Hyperparameters and paths for learned-IK training."""

    # ── Robot ───────────────────────────────────────────────────────────────
    robot: str = "panda"
    """Robot name passed to ``robot_descriptions`` (e.g. ``"panda"``)."""

    target_links: list[str] = dataclasses.field(default_factory=list)
    """Target end-effector link name(s).  Defaults to the preset in
    ROBOT_CONFIGS; must be provided for unknown robots.
    Pass multiple names for multi-EE robots, e.g.
    ``--target_links right_hand left_hand``."""

    fixed_joint_names: list[str] = dataclasses.field(default_factory=list)
    """Joint names that must remain fixed during IK (e.g. finger joints).
    Overrides the preset in ROBOT_CONFIGS when provided."""

    list_links: bool = False
    """If True, print available link names for the robot and exit."""

    # ── Dataset ─────────────────────────────────────────────────────────────
    n_train: int = 2_500_000
    """Number of random training samples (paper: 2.5 M)."""

    n_val: int = 20_000
    """Number of held-out validation samples."""

    # ── Training ────────────────────────────────────────────────────────────
    n_epochs: int = 100
    """Number of training epochs."""

    batch_size: int = 128
    """Mini-batch size (paper: 128)."""

    lr: float = 5e-4
    """Initial learning rate (paper: 5e-4)."""

    seed: int = 42
    """Master random seed."""

    # ── Model ───────────────────────────────────────────────────────────────
    hidden: int = 1024
    """Hidden-layer width of each coupling-layer subnet (default 1024)."""

    n_layers: int = 15
    """Number of affine coupling layers (default 15, matching IKFlow)."""

    latent_dim: int = 15
    """Latent space dimension (≥ n_act; paper uses 15 for 7-8 DOF robots)."""

    # ── SoftFlow ────────────────────────────────────────────────────────────
    softflow_scale: float = 1e-3
    """Maximum noise scale for SoftFlow training (paper: 1e-3).
    Per batch: c ~ U(0, softflow_scale), v ~ N(0,I), q̃ = q + c·v."""

    # ── Output ──────────────────────────────────────────────────────────────
    model_dir: str = ""
    """Directory where the trained model is saved.  Defaults to
    ``resources/learned_ik/`` relative to the project root."""

    verbose: bool = True
    """Print epoch-level training progress."""


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_dataset(
    robot: pk.Robot,
    target_link_indices: list[int],
    fixed_mask: np.ndarray,          # bool, shape (n_act,)
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (pose_encoding, normalised_cfg) pairs.

    For multi-EE robots the pose encoding is the concatenation of per-EE
    12-dim encodings, giving shape ``(n_samples, 12 * n_ee)``.
    The SoftFlow conditioning scalar c is NOT included here; it is added
    per-batch during training.

    Returns:
        pose_enc_np: float32 array, shape ``(n_samples, 12 * n_ee)``.
        cfg_norm_np: float32 array, shape ``(n_samples, n_act)``, in (−1, 1).
    """
    n_act = robot.joints.num_actuated_joints
    n_ee  = len(target_link_indices)
    lower = np.array(robot.joints.lower_limits, dtype=np.float32)
    upper = np.array(robot.joints.upper_limits, dtype=np.float32)
    mid   = (lower + upper) * 0.5
    half  = (upper - lower) * 0.5

    cfgs = rng.uniform(lower, upper, size=(n_samples, n_act)).astype(np.float32)
    cfgs[:, fixed_mask] = mid[fixed_mask]

    CHUNK      = 4096
    all_per_ee = [np.empty((n_samples, 7), dtype=np.float32) for _ in range(n_ee)]
    fk_jit     = jax.jit(jax.vmap(robot.forward_kinematics))
    for start in range(0, n_samples, CHUNK):
        end   = min(start + CHUNK, n_samples)
        chunk = jnp.array(cfgs[start:end])
        Ts    = fk_jit(chunk)
        for ee_i, link_idx in enumerate(target_link_indices):
            all_per_ee[ee_i][start:end] = np.array(Ts[:, link_idx, :])

    @jax.jit
    @jax.vmap
    def _encode_single(wxyz_xyz: jax.Array) -> jax.Array:
        return encode_pose(jaxlie.SE3(wxyz_xyz))

    enc_parts = [
        np.array(_encode_single(jnp.array(all_per_ee[ee_i])), dtype=np.float32)
        for ee_i in range(n_ee)
    ]
    pose_enc_np = np.concatenate(enc_parts, axis=1)   # (n_samples, 12 * n_ee)
    cfg_norm_np = np.clip(((cfgs - mid) / half).astype(np.float32), -1.0, 1.0)

    return pose_enc_np, cfg_norm_np


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    cfg: TrainConfig,
    robot: pk.Robot,
    target_link_indices: list[int],
    target_link_names: list[str],
    fixed_mask: np.ndarray,
    model_dir: Path,
) -> None:
    n_act = robot.joints.num_actuated_joints
    n_ee  = len(target_link_indices)
    rng   = np.random.default_rng(cfg.seed)
    jax_key = jax.random.PRNGKey(cfg.seed)

    # ── Generate data ────────────────────────────────────────────────────
    if cfg.verbose:
        print(f"\nGenerating {cfg.n_train:,} training samples "
              f"(n_ee={n_ee}, pose_dim={12*n_ee}, cond_dim={12*n_ee+1}) ...")
    t0 = time.perf_counter()
    train_enc, train_norm = _generate_dataset(
        robot, target_link_indices, fixed_mask, cfg.n_train, rng,
    )
    if cfg.verbose:
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
        print(f"Generating {cfg.n_val:,} validation samples ...")
    t0 = time.perf_counter()
    val_enc, val_norm = _generate_dataset(
        robot, target_link_indices, fixed_mask, cfg.n_val, rng,
    )
    if cfg.verbose:
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Precompute validation conditioning with c=0 appended (SoftFlow: c=0 at eval)
    val_enc_sf = np.concatenate(
        [val_enc, np.zeros((len(val_enc), 1), dtype=np.float32)], axis=1
    )  # (n_val, 12*n_ee + 1)

    # ── Initialise model ─────────────────────────────────────────────────
    pose_dim  = train_enc.shape[1]     # 12 * n_ee
    cond_dim  = pose_dim + 1           # +1 for SoftFlow scalar c
    model     = IKFlowNet(
        n_act=n_act,
        latent_dim=cfg.latent_dim,
        n_layers=cfg.n_layers,
        hidden=cfg.hidden,
    )
    jax_key, init_key = jax.random.split(jax_key)
    dummy_q    = jnp.zeros(n_act,    dtype=jnp.float32)
    dummy_cond = jnp.zeros(cond_dim, dtype=jnp.float32)  # includes SoftFlow dim
    params  = model.init(init_key, dummy_q, dummy_cond)

    if cfg.verbose:
        n_params = sum(x.size for x in jax.tree.leaves(params))
        print(f"\nModel: IKFlowNet("
              f"n_act={n_act}, latent_dim={cfg.latent_dim}, "
              f"n_layers={cfg.n_layers}, hidden={cfg.hidden}×3, "
              f"cond_dim={cond_dim})"
              f"  [{n_params:,} parameters]")

    # ── Optimiser: RAdam + exponential decay ─────────────────────────────
    # Paper: Ranger (RAdam + Lookahead), LR=5e-4, decay 0.979 every 39 000 steps.
    # Using RAdam here (Lookahead not available in standard optax).
    lr_schedule = optax.exponential_decay(
        init_value=cfg.lr,
        transition_steps=39_000,
        decay_rate=0.979,
        end_value=1e-6,
    )
    tx    = optax.radam(lr_schedule)
    state = flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )

    # ── JIT-compiled train / eval steps ───────────────────────────────────
    # NLL loss: 0.5*||z||² - log_det_J
    # cond already contains the SoftFlow scalar c (appended before calling)

    def _nll_single(params: Any, q_norm: jax.Array, cond: jax.Array) -> jax.Array:
        z, log_det = model.apply(params, q_norm, cond)
        return 0.5 * jnp.sum(z ** 2) - log_det

    _nll_batch = jax.vmap(_nll_single, in_axes=(None, 0, 0))

    @jax.jit
    def train_step(
        state:      flax_train_state.TrainState,
        batch_enc:  jax.Array,   # (B, cond_dim)  — includes SoftFlow c
        batch_norm: jax.Array,   # (B, n_act)      — SoftFlow-noised
    ) -> tuple[flax_train_state.TrainState, jax.Array]:
        def loss_fn(params: Any) -> jax.Array:
            return jnp.mean(_nll_batch(params, batch_norm, batch_enc))
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_loss(
        params:     Any,
        batch_enc:  jax.Array,   # (B, cond_dim) with c=0
        batch_norm: jax.Array,
    ) -> jax.Array:
        return jnp.mean(_nll_batch(params, batch_norm, batch_enc))

    # ── Training loop ────────────────────────────────────────────────────
    indices     = np.arange(cfg.n_train)
    best_val    = float("inf")
    best_params = state.params
    t_start     = time.perf_counter()

    if cfg.verbose:
        print(f"\nTraining {cfg.n_epochs} epochs  "
              f"(batch={cfg.batch_size}, lr={cfg.lr}, "
              f"schedule=exp(0.979/39k), loss=NLL+SoftFlow) ...")

    for epoch in range(cfg.n_epochs):
        rng.shuffle(indices)
        ep_loss   = 0.0
        n_batches = 0

        for start in range(0, cfg.n_train, cfg.batch_size):
            end = min(start + cfg.batch_size, cfg.n_train)
            idx = indices[start:end]
            B   = len(idx)

            # SoftFlow: c ~ U(0, softflow_scale), noise = c * N(0, I)
            c_vals  = rng.uniform(0.0, cfg.softflow_scale, size=(B, 1)).astype(np.float32)
            noise   = rng.standard_normal(size=(B, n_act)).astype(np.float32) * c_vals
            b_norm  = np.clip(train_norm[idx] + noise, -1.0, 1.0)

            # Conditioning: [pose_enc, c]
            b_enc   = np.concatenate([train_enc[idx], c_vals], axis=1)

            state, loss = train_step(state, jnp.array(b_enc), jnp.array(b_norm))
            ep_loss  += float(loss)
            n_batches += 1

        val_loss = float(eval_loss(
            state.params,
            jnp.array(val_enc_sf),
            jnp.array(val_norm),
        ))

        if val_loss < best_val:
            best_val    = val_loss
            best_params = state.params

        if cfg.verbose and ((epoch + 1) % max(1, cfg.n_epochs // 20) == 0 or epoch == 0):
            elapsed = time.perf_counter() - t_start
            print(f"  Epoch {epoch+1:4d}/{cfg.n_epochs}  "
                  f"train={ep_loss/n_batches:.6f}  "
                  f"val={val_loss:.6f}  "
                  f"best={best_val:.6f}  "
                  f"({elapsed:.0f}s)")

    # ── FK-level evaluation (flow sampling, before LM refinement) ─────────
    if cfg.verbose:
        print("\nEvaluating FK error on validation set "
              "(flow sampling, latent_scale=0.25, c=0) ...")

    lower_jax = robot.joints.lower_limits
    upper_jax = robot.joints.upper_limits
    mid_jax   = (lower_jax + upper_jax) * 0.5
    half_jax  = (upper_jax - lower_jax) * 0.5
    fk_jit    = jax.jit(jax.vmap(robot.forward_kinematics))

    n_eval   = min(2048, cfg.n_val)
    # Conditioning at inference: [pose_enc, c=0]
    pred_enc = jnp.array(val_enc_sf[:n_eval])

    jax_key, eval_key = jax.random.split(jax_key)
    # Latent samples scaled by 0.25 (paper inference default)
    zs = jax.random.normal(eval_key, (n_eval, cfg.latent_dim), dtype=jnp.float32) * 0.25

    @jax.jit
    @jax.vmap
    def _flow_sample(z: jax.Array, cond: jax.Array) -> jax.Array:
        q_norm = model.apply(best_params, z, cond, method=IKFlowNet.inverse)
        return jnp.clip(mid_jax + half_jax * q_norm, lower_jax, upper_jax)

    pred_cfgs = _flow_sample(zs, pred_enc)                    # (n_eval, n_act)
    pred_Ts   = np.array(fk_jit(pred_cfgs))                   # (n_eval, n_links, 7)

    true_cfgs = jnp.clip(
        mid_jax + half_jax * jnp.array(val_norm[:n_eval]),
        lower_jax, upper_jax,
    )
    true_Ts = np.array(fk_jit(true_cfgs))                     # (n_eval, n_links, 7)

    @jax.jit
    @jax.vmap
    def _pose_err(pred_T: jax.Array, true_T: jax.Array) -> tuple[jax.Array, jax.Array]:
        log = (jaxlie.SE3(pred_T).inverse() @ jaxlie.SE3(true_T)).log()
        return jnp.linalg.norm(log[:3]), jnp.linalg.norm(log[3:])

    for ee_i, (link_idx, link_name) in enumerate(
        zip(target_link_indices, target_link_names)
    ):
        pos_e, rot_e = _pose_err(
            jnp.array(pred_Ts[:, link_idx, :]),
            jnp.array(true_Ts[:, link_idx, :]),
        )
        if cfg.verbose:
            print(f"  EE '{link_name}': "
                  f"pos_med={float(jnp.median(pos_e))*1e3:.2f} mm  "
                  f"rot_med={float(jnp.median(rot_e)):.4f} rad")

    if cfg.verbose:
        print("  (LM refinement at inference reduces errors further)")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = model_dir / f"{cfg.robot}.pkl"
    save_learned_ik(
        path                = save_path,
        params              = best_params,
        robot_name          = cfg.robot,
        target_link_names   = target_link_names,
        target_link_indices = target_link_indices,
        n_act               = n_act,
        latent_dim          = cfg.latent_dim,
    )
    if cfg.verbose:
        print(f"\nModel saved → {save_path}")
        print(f"Best validation loss: {best_val:.6f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: TrainConfig) -> None:
    # ── Resolve robot config ──────────────────────────────────────────────
    preset       = ROBOT_CONFIGS.get(cfg.robot, {})
    target_links = cfg.target_links or preset.get("target_links", [])
    fixed_names  = cfg.fixed_joint_names or preset.get("fixed_joint_names", [])

    # ── Load robot ────────────────────────────────────────────────────────
    print(f"Loading robot '{cfg.robot}' from robot_descriptions ...")
    urdf  = load_robot_description(f"{cfg.robot}_description")
    robot = pk.Robot.from_urdf(urdf)
    n_act = robot.joints.num_actuated_joints

    print(f"  {n_act} actuated joints")
    print(f"  Actuated joints: {list(robot.joints.actuated_names)}")

    # ── List links mode ───────────────────────────────────────────────────
    if cfg.list_links:
        print("\nAvailable link names:")
        for i, name in enumerate(robot.links.names):
            print(f"  [{i:3d}] {name}")
        return

    # ── Resolve and validate target links ────────────────────────────────
    if not target_links:
        raise ValueError(
            f"No default target links known for robot '{cfg.robot}'.  "
            f"Provide them with --target_links <name> [<name2> ...].  "
            f"Run with --list_links to see available link names."
        )

    target_link_indices: list[int] = []
    for link_name in target_links:
        if link_name not in robot.links.names:
            raise ValueError(
                f"Link '{link_name}' not found in robot '{cfg.robot}'.  "
                f"Run with --list_links to see available link names."
            )
        target_link_indices.append(robot.links.names.index(link_name))

    if cfg.latent_dim < n_act:
        raise ValueError(
            f"latent_dim ({cfg.latent_dim}) must be >= n_act ({n_act})."
        )

    fixed_mask = np.array(
        [name in fixed_names for name in robot.joints.actuated_names], dtype=bool
    )

    n_ee = len(target_links)
    print(f"\n  End-effectors ({n_ee}):")
    for name, idx in zip(target_links, target_link_indices):
        print(f"    '{name}' (link index {idx})")
    print(f"  Fixed joints: {[n for n in fixed_names if n in robot.joints.actuated_names]}")
    print(f"  Conditioning dim: {12*n_ee} pose + 1 SoftFlow = {12*n_ee+1}")
    print(f"  Latent dim: {cfg.latent_dim}  (n_act={n_act}, pad={cfg.latent_dim - n_act})")

    # ── Output directory ─────────────────────────────────────────────────
    if cfg.model_dir:
        model_dir = Path(cfg.model_dir)
    else:
        model_dir = get_default_model_path(cfg.robot).parent

    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    _train(cfg, robot, target_link_indices, target_links, fixed_mask, model_dir)


if __name__ == "__main__":
    tyro.cli(main)
