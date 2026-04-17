"""StompMotionGenerator: Cartesian spline seeding + IK + STOMP/MPPI trajopt.

Full pipeline:
  1. Cartesian spline interpolation between control SE(3) poses (configurable
     mode: linear, cubic, bspline).
  2. Batched IK via MPPI to convert waypoint poses to joint configs.
  3. STOMP/MPPI trajectory optimization on the resulting batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from .._splines import (
    SplineMode,
    bspline_interpolate,
    cubic_spline_interpolate,
    linear_interpolate,
)
from ..collision import RobotCollision
from ..optimization_engines._mppi_ik import mppi_ik_solve_cuda_batch
from ..optimization_engines._stomp_optimization import StompTrajOptConfig, stomp_trajopt
from ._trajopt import IKSeedConfig


@dataclass
class StompMotionGenerator:
    """Motion generator that plans in Cartesian space, seeds via IK, and
    refines with STOMP/MPPI trajectory optimization.

    Pipeline:
      1. Build Cartesian control poses: ``[start_pose, *waypoint_poses, goal_pose]``.
      2. Spline-interpolate the position and orientation components to produce
         ``n_timesteps`` SE(3) waypoint poses.
      3. Batched MPPI CUDA IK to solve all T IK problems in one kernel launch.
      4. Tile the base trajectory to ``[B, T, DOF]`` and add noise.
      5. Run ``stomp_trajopt`` on the batch.

    Args:
        robot:                 Robot kinematics pytree.
        robot_coll:            Robot collision model pytree.
        world_geoms:           Tuple of stacked world collision geometry objects.
        ee_link_name:          Name of the end-effector link for Cartesian IK.
        n_timesteps:           Number of waypoints per trajectory.
        n_batch:               Number of candidate trajectories in the batch.
        noise_scale:           Std of per-seed Gaussian perturbation for IK seeding.
        cartesian_spline_mode: Spline mode for Cartesian interpolation.
        trajopt_cfg:           STOMP trajectory optimization configuration.
        ik_cfg:                IK seeding configuration.
        use_cuda:              Whether to use the CUDA backend for trajopt.
    """

    robot: Robot
    robot_coll: RobotCollision
    world_geoms: tuple
    ee_link_name: str

    n_timesteps: int = 64
    n_batch: int = 25
    noise_scale: float = 0.05
    cartesian_spline_mode: SplineMode = "linear"

    trajopt_cfg: StompTrajOptConfig = field(default_factory=StompTrajOptConfig)
    ik_cfg: IKSeedConfig = field(default_factory=IKSeedConfig)
    use_cuda: bool = False

    @property
    def _ee_idx(self) -> int:
        return self.robot.links.names.index(self.ee_link_name)

    def _cartesian_spline_interpolation(
        self,
        control_poses: jaxlie.SE3,
    ) -> jaxlie.SE3:
        """Spline-interpolate K control SE(3) poses to n_timesteps waypoints."""
        positions = control_poses.translation()
        rotations = control_poses.rotation()

        mode = self.cartesian_spline_mode
        if mode == "linear":
            interp_pos = linear_interpolate(positions, self.n_timesteps)
        elif mode == "cubic":
            interp_pos = cubic_spline_interpolate(positions, self.n_timesteps)
        elif mode == "bspline":
            degree = min(3, positions.shape[0] - 1)
            interp_pos = bspline_interpolate(positions, self.n_timesteps, degree=degree)
        else:
            raise ValueError(f"Unknown spline mode: {mode!r}")

        start_rot = jaxlie.SO3(rotations.wxyz[0:1])
        rel_rots  = start_rot.inverse() @ rotations
        log_vecs  = rel_rots.log()

        if mode == "linear":
            interp_logs = linear_interpolate(log_vecs, self.n_timesteps)
        elif mode == "cubic":
            interp_logs = cubic_spline_interpolate(log_vecs, self.n_timesteps)
        elif mode == "bspline":
            degree = min(3, log_vecs.shape[0] - 1)
            interp_logs = bspline_interpolate(log_vecs, self.n_timesteps, degree=degree)

        interp_rot = jaxlie.SO3(start_rot.wxyz[0]) @ jaxlie.SO3.exp(interp_logs)
        return jaxlie.SE3.from_rotation_and_translation(interp_rot, interp_pos)

    def _batch_ik(
        self,
        interp_poses: jaxlie.SE3,
        prev_cfgs: Float[Array, "T DOF"],
        key: Array,
    ) -> Float[Array, "T DOF"]:
        """Solve T IK problems via batched MPPI (no collision avoidance)."""
        ik_cfg = self.ik_cfg
        return mppi_ik_solve_cuda_batch(
            robot=self.robot,
            target_link_indices=self._ee_idx,
            target_poses=interp_poses,
            rng_key=key,
            previous_cfgs=prev_cfgs,
            num_seeds=ik_cfg.num_seeds,
            n_particles=ik_cfg.n_particles,
            n_mppi_iters=ik_cfg.n_mppi_iters,
            n_lbfgs_iters=ik_cfg.n_lbfgs_iters,
            m_lbfgs=ik_cfg.m_lbfgs,
            pos_weight=ik_cfg.pos_weight,
            ori_weight=ik_cfg.ori_weight,
            sigma=ik_cfg.sigma,
            mppi_temperature=ik_cfg.mppi_temperature,
            eps_pos=ik_cfg.eps_pos,
            eps_ori=ik_cfg.eps_ori,
            continuity_weight=ik_cfg.continuity_weight,
        )

    def _seed_trajectories(
        self,
        control_poses: jaxlie.SE3,
        key: Array,
        prev_cfgs: Float[Array, "T DOF"] | None = None,
    ) -> tuple[Float[Array, "B T DOF"], Float[Array, "DOF"], Float[Array, "DOF"]]:
        """Build [B, T, DOF] batch via Cartesian spline + IK seeding + noise."""
        key, ik_key = jax.random.split(key)

        interp_poses = self._cartesian_spline_interpolation(control_poses)

        if prev_cfgs is None:
            mid_cfg = (
                self.robot.joints.lower_limits + self.robot.joints.upper_limits
            ) / 2.0
            prev_cfgs = jnp.broadcast_to(
                mid_cfg[None], (self.n_timesteps, mid_cfg.shape[0])
            )

        base_traj = self._batch_ik(interp_poses, prev_cfgs, ik_key)

        start_cfg = base_traj[0]
        goal_cfg  = base_traj[-1]

        trajs = jnp.broadcast_to(
            base_traj[None], (self.n_batch, self.n_timesteps, base_traj.shape[-1])
        )
        noise = jax.random.normal(key, trajs.shape) * self.noise_scale
        return trajs + noise, start_cfg, goal_cfg

    def generate(
        self,
        start_pose: jaxlie.SE3,
        goal_pose: jaxlie.SE3,
        key: Array,
        waypoint_poses: jaxlie.SE3 | None = None,
        prev_cfgs: Float[Array, "T DOF"] | None = None,
    ) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"], float, float]:
        """Run the full pipeline: Cartesian spline → IK seeding → STOMP trajopt.

        Args:
            start_pose:     Start end-effector SE(3) pose.
            goal_pose:      Goal end-effector SE(3) pose.
            key:            JAX PRNG key.
            waypoint_poses: Optional intermediate SE(3) waypoint poses.
            prev_cfgs:      Optional warm-start configs for IK. Shape [T, DOF].

        Returns:
            best_traj:     Best trajectory by nonlinear cost. Shape [T, DOF].
            costs:         Final cost per trajectory.          Shape [B].
            final_trajs:   All optimized trajectories.         Shape [B, T, DOF].
            trajopt_time:  Wall-clock time for the STOMP trajopt step (seconds).
            ik_time:       Wall-clock time for the IK seeding step (seconds).
        """
        start_wxyz_xyz = start_pose.wxyz_xyz[None]
        goal_wxyz_xyz  = goal_pose.wxyz_xyz[None]

        if waypoint_poses is not None:
            wp_wxyz_xyz = waypoint_poses.wxyz_xyz
            if wp_wxyz_xyz.ndim == 1:
                wp_wxyz_xyz = wp_wxyz_xyz[None]
            all_wxyz_xyz = jnp.concatenate(
                [start_wxyz_xyz, wp_wxyz_xyz, goal_wxyz_xyz], axis=0
            )
        else:
            all_wxyz_xyz = jnp.concatenate(
                [start_wxyz_xyz, goal_wxyz_xyz], axis=0
            )

        control_poses = jaxlie.SE3(all_wxyz_xyz)

        key, stomp_key = jax.random.split(key)

        t0_ik = time.perf_counter()
        init_trajs, start_cfg, goal_cfg = self._seed_trajectories(
            control_poses, key, prev_cfgs=prev_cfgs,
        )
        ik_time = time.perf_counter() - t0_ik

        t0 = time.perf_counter()
        best_traj, costs, final_trajs = stomp_trajopt(
            init_trajs,
            start_cfg,
            goal_cfg,
            self.robot,
            self.robot_coll,
            self.world_geoms,
            self.trajopt_cfg,
            key=stomp_key,
            use_cuda=self.use_cuda,
        )
        best_traj.block_until_ready()
        trajopt_time = time.perf_counter() - t0

        return best_traj, costs, final_trajs, trajopt_time, ik_time
