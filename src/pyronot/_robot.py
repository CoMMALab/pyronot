from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from ._robot_urdf_parser import JointInfo, LinkInfo, RobotURDFParser


@jdc.pytree_dataclass
class Robot:
    """A differentiable robot kinematics tree."""

    joints: JointInfo
    """Joint information for the robot."""

    links: LinkInfo
    """Link information for the robot."""

    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> Robot:
        """
        Loads a robot kinematic tree from a URDF.
        Internally tracks a topological sort of the joints.

        Args:
            urdf: The URDF to load the robot from.
            default_joint_cfg: The default joint configuration to use for optimization.
        """
        joints, links = RobotURDFParser.parse(urdf)

        # Compute default joint configuration.
        if default_joint_cfg is None:
            default_joint_cfg = (joints.lower_limits + joints.upper_limits) / 2
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)
        assert default_joint_cfg.shape == (joints.num_actuated_joints,)

        # Variable class for the robot configuration.
        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_joint_cfg,
        ): ...

        robot = Robot(
            joints=joints,
            links=links,
            joint_var_cls=JointVar,
        )

        return robot

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
        use_cuda: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch link_count 7"]:
        """Run forward kinematics on the robot's links, in the provided configuration.

        Computes the world pose of each link frame. The result is ordered
        corresponding to `self.link.names`.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.
            unroll_fk: If True, unroll the JAX fori_loop over joints (ignored when use_cuda=True).
            use_cuda: If True, dispatch to an external CUDA kernel via the JAX FFI instead of
                the default JAX implementation.  Requires ``_fk_cuda.so`` to be compiled first
                (see ``src/pyronot/cuda_kernels/build_fk_cuda.sh``).

        Returns:
            The SE(3) transforms of the links, ordered by `self.link.names`,
            in the format `(*batch, link_count, wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)

        if use_cuda:
            from .cuda_kernels._fk_cuda import fk_cuda
            Ts_world_joint = fk_cuda(
                cfg=cfg,
                twists=self.joints.twists,
                parent_tf=self.joints.parent_transforms,
                parent_idx=self.joints.parent_indices,
                act_idx=self.joints.actuated_indices,
                mimic_mul=self.joints.mimic_multiplier,
                mimic_off=self.joints.mimic_offset,
                mimic_act_idx=self.joints.mimic_act_indices,
                topo_inv=self.joints._topo_sort_inv,
                fk_level_starts=self.joints.fk_level_starts,
                fk_level_joints=self.joints.fk_level_joints,
            )
        else:
            Ts_world_joint = self._forward_kinematics_joints(cfg, unroll_fk)

        return self._link_poses_from_joint_poses(Ts_world_joint)

    @jdc.jit
    def inverse_kinematics(
        self,
        target_link_name: jdc.Static[str],
        target_pose: jaxlie.SE3,
        rng_key: Array | None = None,
        previous_cfg: Float[Array, "n_actuated_joints"] | None = None,
        num_seeds: jdc.Static[int] = 32,
        coarse_max_iter: jdc.Static[int] = 20,
        lm_max_iter: jdc.Static[int] = 40,
        epsilon: float = 0.02,
        nu: float = float(jnp.pi / 2),
        lambda_init: float = 5e-3,
        continuity_weight: float = 1e-3,
        fixed_joint_mask: Float[Array, "n_actuated_joints"] | None = None,
    ) -> Float[Array, "n_actuated_joints"]:
        """Solve inverse kinematics using the HJCD-IK two-phase optimizer.

        Phase 1 samples *num_seeds* configurations — the first ``top_k`` are
        warm-started near *previous_cfg* (or the joint-range midpoint when not
        provided) and the rest are random — then refines them via greedy
        coordinate descent.  Phase 2 selects the best solutions and polishes
        them with Levenberg-Marquardt.  A small *continuity_weight* penalty on
        distance from *previous_cfg* is added to the final selection criterion
        to stabilise the choice between equally valid IK solutions.

        Args:
            target_link_name:  Name of the link whose pose should match *target_pose*.
            target_pose:       Desired SE(3) world pose for that link.
            rng_key:           JAX PRNG key (defaults to PRNGKey(0) if None).
            previous_cfg:      Previous joint configuration for warm-starting and
                               continuity-aware selection.  Defaults to joint-range
                               midpoint when not provided.
            num_seeds:         Number of random seeds for the coarse phase.
            coarse_max_iter:   Coordinate-descent iteration budget.
            lm_max_iter:       Levenberg-Marquardt iteration budget.
            epsilon:           Position convergence threshold [m] (20 mm).
            nu:                Orientation convergence threshold [rad] (π/2).
            lambda_init:       Initial LM damping factor.
            continuity_weight: Weight on ‖q − previous_cfg‖² in best-solution
                               selection (default 1e-3).

        Returns:
            Best joint configuration found, shape ``(n_actuated_joints,)``.
        """
        from .optimization_engines._hjcd_ik import hjcd_solve

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        if previous_cfg is None:
            previous_cfg = (self.joints.lower_limits + self.joints.upper_limits) / 2

        target_link_index = self.links.names.index(target_link_name)
        return hjcd_solve(
            robot=self,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            num_seeds=num_seeds,
            coarse_max_iter=coarse_max_iter,
            lm_max_iter=lm_max_iter,
            epsilon=epsilon,
            nu=nu,
            lambda_init=lambda_init,
            continuity_weight=continuity_weight,
            fixed_joint_mask=fixed_joint_mask,
        )

    def _link_poses_from_joint_poses(
        self, Ts_world_joint: Float[Array, "*batch actuated_count 7"]
    ) -> Float[Array, "*batch link_count 7"]:
        (*batch_axes, _, _) = Ts_world_joint.shape
        # Get the link poses.
        base_link_mask = self.links.parent_joint_indices == -1
        parent_joint_indices = jnp.where(
            base_link_mask, 0, self.links.parent_joint_indices
        )
        identity_pose = jaxlie.SE3.identity().wxyz_xyz
        Ts_world_link = jnp.where(
            base_link_mask[..., None],
            identity_pose,
            Ts_world_joint[..., parent_joint_indices, :],
        )
        assert Ts_world_link.shape == (*batch_axes, self.links.num_links, 7)
        return Ts_world_link

    def _forward_kinematics_joints(
        self,
        cfg: Float[Array, "*batch actuated_count"],
        unroll_fk: jdc.Static[bool] = False,
    ) -> Float[Array, "*batch joint_count 7"]:
        (*batch_axes, _) = cfg.shape
        assert cfg.shape == (*batch_axes, self.joints.num_actuated_joints)

        # Calculate full configuration using the dedicated method
        q_full = self.joints.get_full_config(cfg)

        # Calculate delta transforms using the effective config and twists for all joints.
        tangents = self.joints.twists * q_full[..., None]
        assert tangents.shape == (*batch_axes, self.joints.num_joints, 6)
        delta_Ts = jaxlie.SE3.exp(tangents)  # Shape: (*batch_axes, self.joint.count, 7)

        # Combine constant parent transform with variable joint delta transform.
        Ts_parent_child = (
            jaxlie.SE3(self.joints.parent_transforms) @ delta_Ts
        ).wxyz_xyz
        assert Ts_parent_child.shape == (*batch_axes, self.joints.num_joints, 7)

        # Topological sort helpers
        topo_order = jnp.argsort(self.joints._topo_sort_inv)
        Ts_parent_child_sorted = Ts_parent_child[..., self.joints._topo_sort_inv, :]
        parent_orig_for_sorted_child = self.joints.parent_indices[
            self.joints._topo_sort_inv
        ]
        idx_parent_joint_sorted = jnp.where(
            parent_orig_for_sorted_child == -1,
            -1,
            topo_order[parent_orig_for_sorted_child],
        )

        # Compute link transforms relative to world, indexed by sorted *joint* index.
        def compute_transform(i: int, Ts_world_link_sorted: Array) -> Array:
            parent_sorted_idx = idx_parent_joint_sorted[i]
            T_world_parent_link = jnp.where(
                parent_sorted_idx == -1,
                jaxlie.SE3.identity().wxyz_xyz,
                Ts_world_link_sorted[..., parent_sorted_idx, :],
            )
            return Ts_world_link_sorted.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent_link)
                    @ jaxlie.SE3(Ts_parent_child_sorted[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_link_init_sorted = jnp.zeros((*batch_axes, self.joints.num_joints, 7))
        Ts_world_link_sorted = jax.lax.fori_loop(
            lower=0,
            upper=self.joints.num_joints,
            body_fun=compute_transform,
            init_val=Ts_world_link_init_sorted,
            unroll=unroll_fk,
        )

        Ts_world_link_joint_indexed = Ts_world_link_sorted[..., topo_order, :]
        assert Ts_world_link_joint_indexed.shape == (
            *batch_axes,
            self.joints.num_joints,
            7,
        )  # This is the link poses indexed by parent *joint* index.

        return Ts_world_link_joint_indexed
