"""Diagnostic script to pinpoint JAX vs CUDA FK divergence per joint."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description


def _fk_joints_jax(robot, cfg):
    """Return (n_joints, 7) world transforms via JAX, original-joint indexed."""
    return np.array(robot._forward_kinematics_joints(cfg))


def _fk_joints_cuda(robot, cfg):
    """Return (n_joints, 7) world transforms via CUDA kernel directly."""
    from pyroffi.cuda_kernels._fk_cuda import fk_cuda
    return np.array(fk_cuda(
        cfg=cfg,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
    ))


def diagnose(robot_name: str, batch: int = 1, seed: int = 42):
    print(f"\n{'='*70}")
    print(f"Diagnosing {robot_name} robot  (batch={batch})")
    print(f"{'='*70}")

    urdf = load_robot_description(f"{robot_name}_description")
    robot = pk.Robot.from_urdf(urdf)

    rng = np.random.default_rng(seed)
    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)
    cfg_np = rng.uniform(lo, hi, size=(batch, robot.joints.num_actuated_joints)).astype(np.float32)
    cfg = jnp.array(cfg_np)

    # JIT warm-up
    fk_jax_jit  = jax.jit(robot._forward_kinematics_joints)
    fk_cuda_raw = jax.jit(lambda c: _fk_joints_cuda_jit(robot, c))

    from pyroffi.cuda_kernels._fk_cuda import fk_cuda
    fk_cuda_jit = jax.jit(lambda c: fk_cuda(
        cfg=c,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
    ))

    # Warm-up
    for _ in range(3):
        jax.block_until_ready(fk_jax_jit(cfg))
        jax.block_until_ready(fk_cuda_jit(cfg))

    jax_out  = np.array(fk_jax_jit(cfg))   # (batch, n_joints, 7)
    cuda_out = np.array(fk_cuda_jit(cfg))  # (batch, n_joints, 7)

    topo_inv = np.array(robot.joints._topo_sort_inv)
    joint_names = robot.joints.names
    parent_idx  = np.array(robot.joints.parent_indices)
    act_idx     = np.array(robot.joints.actuated_indices)
    twists      = np.array(robot.joints.twists)  # (n_joints, 6)

    def _joint_type(twist: np.ndarray) -> str:
        """Infer joint type from twist vector: linear part = twist[:3], angular = twist[3:]."""
        lin = np.abs(twist[:3]).max()
        ang = np.abs(twist[3:]).max()
        if lin < 1e-9 and ang < 1e-9:
            return "fixed"
        if ang >= 1e-9 and lin < 1e-9:
            return "revolute"
        if lin >= 1e-9 and ang < 1e-9:
            return "prismatic"
        return "other"

    print(f"\nJoint-level comparison  (batch element 0):")
    print(f"{'Sorted':>6}  {'Orig':>4}  {'Name':<35}  {'Type':<10}  {'act':>4}  {'par':>4}  {'Max|err|':>10}  Note")
    print("-" * 97)

    for si in range(robot.joints.num_joints):
        orig_j = int(topo_inv[si])
        err = float(np.abs(jax_out[0, orig_j] - cuda_out[0, orig_j]).max())

        jax_q  = jax_out[0, orig_j, :4]
        cuda_q = cuda_out[0, orig_j, :4]
        jax_t  = jax_out[0, orig_j, 4:]
        cuda_t = cuda_out[0, orig_j, 4:]
        t_err  = float(np.abs(jax_t - cuda_t).max())
        q_err  = float(np.abs(jax_q - cuda_q).max())

        jtype = _joint_type(twists[orig_j])

        note = ""
        if err > 1e-4:
            # Check if it's a quaternion sign flip
            if float(np.abs(jax_q + cuda_q).max()) < 1e-3:
                note = "QUAT SIGN FLIP"
            elif t_err > 1e-4:
                note = f"TRANS err={t_err:.3e}"
            elif q_err > 1e-4:
                note = f"QUAT err={q_err:.3e}"
            flag = " <-- FAIL"
        else:
            flag = ""

        print(f"{si:>6}  {orig_j:>4}  {joint_names[orig_j]:<35}  "
              f"{jtype:<10}  {act_idx[orig_j]:>4}  {parent_idx[orig_j]:>4}  "
              f"{err:>10.3e}  {note}{flag}")

    # Print the q_full for batch element 0
    print(f"\nFull config (q_full[0]) from get_full_config:")
    q_full = np.array(robot.joints.get_full_config(cfg[0:1]))[0]
    print(" ".join(f"{v:7.4f}" for v in q_full))

    print(f"\nActuated cfg[0]:")
    print(" ".join(f"{v:7.4f}" for v in cfg_np[0]))

    # Also compare full forward_kinematics (link poses)
    fk_jax_full  = jax.jit(lambda c: robot.forward_kinematics(c, use_cuda=False))
    fk_cuda_full = jax.jit(lambda c: robot.forward_kinematics(c, use_cuda=True))
    for _ in range(3):
        jax.block_until_ready(fk_jax_full(cfg))
        jax.block_until_ready(fk_cuda_full(cfg))
    jax_links  = np.array(fk_jax_full(cfg))
    cuda_links = np.array(fk_cuda_full(cfg))

    print(f"\nLink-level comparison (batch element 0):")
    print(f"{'Idx':>4}  {'Name':<35}  {'par_jnt':>7}  {'Max|err|':>10}  Note")
    print("-" * 70)
    par_jnt_idx = np.array(robot.links.parent_joint_indices)
    for li, lname in enumerate(robot.links.names):
        err = float(np.abs(jax_links[0, li] - cuda_links[0, li]).max())
        jl = jax_links[0, li, :4]; cl = cuda_links[0, li, :4]
        note = ""
        if err > 1e-4:
            t_err = float(np.abs(jax_links[0, li, 4:] - cuda_links[0, li, 4:]).max())
            if float(np.abs(jl + cl).max()) < 1e-3:
                note = "QUAT SIGN FLIP"
            elif t_err > 1e-4:
                note = f"TRANS err={t_err:.3e}"
            else:
                note = f"QUAT err"
            flag = " <-- FAIL"
        else:
            flag = ""
        print(f"{li:>4}  {lname:<35}  {par_jnt_idx[li]:>7}  {err:>10.3e}  {note}{flag}")

    print(f"\ntopo_sort_inv: {topo_inv.tolist()}")
    print(f"parent_indices: {parent_idx.tolist()}")


if __name__ == "__main__":
    diagnose("panda",  batch=1)
    diagnose("baxter", batch=1)
    diagnose("baxter", batch=256)
