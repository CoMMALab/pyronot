"""TrajoptMotionGenerator: Cartesian spline seeding + collision-free IK + SCO trajopt.

Full pipeline:
  1. Cartesian spline interpolation between control SE(3) poses (configurable
     mode: linear, cubic, bspline).
  2. Batched collision-free IK via MPPI to convert waypoint poses to joint configs.
  3. SCO trajectory optimization on the resulting batch.
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
from ..collision import RobotCollision, colldist_from_sdf
from ..optimization_engines._mppi_ik import mppi_ik_solve_cuda_batch
from ..optimization_engines._sco_optimization import TrajOptConfig, sco_trajopt


@dataclass(frozen=True)
class IKSeedConfig:
    """Configuration for the collision-free IK seeding stage."""

    num_seeds: int = 8
    n_particles: int = 64
    n_mppi_iters: int = 20
    n_lbfgs_iters: int = 25
    m_lbfgs: int = 5
    collision_weight: float = 1e4
    collision_margin: float = 0.01


@dataclass
class TrajoptMotionGenerator:
    """Motion generator that plans in Cartesian space, seeds via collision-free
    IK, and refines with SCO trajectory optimization.

    Pipeline:
      1. Build Cartesian control poses: ``[start_pose, *waypoint_poses, goal_pose]``.
      2. Spline-interpolate the position and orientation components to produce
         ``n_timesteps`` SE(3) waypoint poses.  The ``cartesian_spline_mode``
         parameter selects the interpolation method (linear, cubic, bspline).
      3. Batched MPPI CUDA IK with self + world collision penalties to solve
         all T IK problems in one kernel launch.
      4. Tile the base trajectory to ``[B, T, DOF]`` and add noise.
      5. Run ``sco_trajopt`` on the batch.

    Args:
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        ee_link_name: Name of the end-effector link for Cartesian IK.
        n_timesteps: Number of waypoints per trajectory.
        n_batch:     Number of candidate trajectories in the batch.
        noise_scale: Std of per-seed Gaussian perturbation.
        cartesian_spline_mode: Spline mode for Cartesian interpolation.
        trajopt_cfg: SCO trajectory optimization configuration.
        ik_cfg:      Collision-free IK seeding configuration.
        use_cuda:    Whether to use the CUDA backend for trajopt.
    """

    robot: Robot
    robot_coll: RobotCollision
    world_geoms: tuple
    ee_link_name: str

    n_timesteps: int = 64
    n_batch: int = 25
    noise_scale: float = 0.05
    cartesian_spline_mode: SplineMode = "linear"

    trajopt_cfg: TrajOptConfig = field(default_factory=TrajOptConfig)
    ik_cfg: IKSeedConfig = field(default_factory=IKSeedConfig)
    use_cuda: bool = False

    @property
    def _ee_idx(self) -> int:
        return self.robot.links.names.index(self.ee_link_name)

    def _cartesian_spline_interpolation(
        self,
        control_poses: jaxlie.SE3,
    ) -> jaxlie.SE3:
        """Spline-interpolate K control SE(3) poses to n_timesteps waypoints.

        Position components are interpolated with the chosen spline mode.
        Orientation components are interpolated via spline in the SO(3) tangent
        space relative to the first control pose's rotation.
        """
        positions = control_poses.translation()       # [K, 3]
        rotations = control_poses.rotation()

        # Position spline: [K, 3] → [T, 3]
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

        # Orientation spline in SO(3) tangent space relative to start rotation.
        # log_k = log(R_start^-1 @ R_k) gives [K, 3] tangent vectors with log_0 = 0.
        start_rot = jaxlie.SO3(rotations.wxyz[0:1])  # broadcast-friendly [1, 4]
        rel_rots = start_rot.inverse() @ rotations    # [K]
        log_vecs = rel_rots.log()                      # [K, 3]

        if mode == "linear":
            interp_logs = linear_interpolate(log_vecs, self.n_timesteps)
        elif mode == "cubic":
            interp_logs = cubic_spline_interpolate(log_vecs, self.n_timesteps)
        elif mode == "bspline":
            degree = min(3, log_vecs.shape[0] - 1)
            interp_logs = bspline_interpolate(log_vecs, self.n_timesteps, degree=degree)

        # Recover rotations: R_t = R_start @ exp(interp_log_t)
        interp_rot = jaxlie.SO3(start_rot.wxyz[0]) @ jaxlie.SO3.exp(interp_logs)  # [T]

        return jaxlie.SE3.from_rotation_and_translation(interp_rot, interp_pos)

    def _collision_free_ik(
        self,
        interp_poses: jaxlie.SE3,
        prev_cfgs: Float[Array, "T DOF"],
        key: Array,
    ) -> Float[Array, "T DOF"]:
        """Solve T IK problems with collision avoidance via batched MPPI."""
        robot_coll = self.robot_coll
        world_geoms = self.world_geoms
        ik_cfg = self.ik_cfg

        def _collision_penalty(cfg, robot, _):
            cost = jnp.zeros(())
            self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
            cost += jnp.sum(
                -jnp.minimum(colldist_from_sdf(self_dists, ik_cfg.collision_margin), 0.0)
            )
            for wg in world_geoms:
                wd = robot_coll.compute_world_collision_distance(robot, cfg, wg)
                cost += jnp.sum(
                    -jnp.minimum(colldist_from_sdf(wd, ik_cfg.collision_margin), 0.0)
                )
            return cost

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
            constraints=[_collision_penalty],
            constraint_args=[jnp.zeros(())],
            constraint_weights=[ik_cfg.collision_weight],
        )

    def _seed_trajectories(
        self,
        control_poses: jaxlie.SE3,
        key: Array,
        prev_cfgs: Float[Array, "T DOF"] | None = None,
    ) -> tuple[Float[Array, "B T DOF"], Float[Array, "DOF"], Float[Array, "DOF"]]:
        """Build [B, T, DOF] batch via Cartesian spline + IK seeding + noise.

        Returns:
            init_trajs: Batch of seeded trajectories. Shape [B, T, DOF].
            start_cfg:  IK solution at the first waypoint.  Shape [DOF].
            goal_cfg:   IK solution at the last waypoint.   Shape [DOF].
        """
        key, ik_key = jax.random.split(key)

        interp_poses = self._cartesian_spline_interpolation(control_poses)

        # Default warm-start: middle-of-range config broadcast to all timesteps
        if prev_cfgs is None:
            mid_cfg = (self.robot.joints.lower_limits + self.robot.joints.upper_limits) / 2.0
            prev_cfgs = jnp.broadcast_to(
                mid_cfg[None], (self.n_timesteps, mid_cfg.shape[0])
            )

        base_traj = self._collision_free_ik(interp_poses, prev_cfgs, ik_key)

        # Extract start/goal joint configs from the IK solutions
        start_cfg = base_traj[0]
        goal_cfg = base_traj[-1]

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
    ) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"], float]:
        """Run the full pipeline: Cartesian spline → IK seeding → SCO trajopt.

        Args:
            start_pose:     Start end-effector SE(3) pose.
            goal_pose:      Goal end-effector SE(3) pose.
            key:            JAX PRNG key.
            waypoint_poses: Optional intermediate SE(3) waypoint poses.  The
                            full control sequence is
                            ``[start_pose, *waypoint_poses, goal_pose]``.
            prev_cfgs:      Optional warm-start configs for IK. Shape [T, DOF].
                            If None, uses mid-range config broadcast.

        Returns:
            best_traj:      Best trajectory by nonlinear cost. Shape [T, DOF].
            costs:          Final cost per trajectory.          Shape [B].
            final_trajs:    All optimized trajectories.         Shape [B, T, DOF].
            trajopt_time:   Wall-clock time for the SCO trajopt step (seconds).
        """
        # Build control poses sequence: [start, *waypoints, goal]
        start_wxyz_xyz = start_pose.wxyz_xyz[None]  # [1, 7]
        goal_wxyz_xyz = goal_pose.wxyz_xyz[None]     # [1, 7]

        if waypoint_poses is not None:
            wp_wxyz_xyz = waypoint_poses.wxyz_xyz     # [W, 7] or [7]
            if wp_wxyz_xyz.ndim == 1:
                wp_wxyz_xyz = wp_wxyz_xyz[None]       # [1, 7]
            all_wxyz_xyz = jnp.concatenate(
                [start_wxyz_xyz, wp_wxyz_xyz, goal_wxyz_xyz], axis=0
            )
        else:
            all_wxyz_xyz = jnp.concatenate(
                [start_wxyz_xyz, goal_wxyz_xyz], axis=0
            )

        control_poses = jaxlie.SE3(all_wxyz_xyz)

        init_trajs, start_cfg, goal_cfg = self._seed_trajectories(
            control_poses, key, prev_cfgs=prev_cfgs,
        )
        t0 = time.perf_counter()
        best_traj, costs, final_trajs = sco_trajopt(
            init_trajs,
            start_cfg,
            goal_cfg,
            self.robot,
            self.robot_coll,
            self.world_geoms,
            self.trajopt_cfg,
            use_cuda=self.use_cuda,
        )
        best_traj.block_until_ready()
        trajopt_time = time.perf_counter() - t0
        return best_traj, costs, final_trajs, trajopt_time
