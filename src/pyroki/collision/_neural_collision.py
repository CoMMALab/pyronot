from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot
    from ._geometry import CollGeom

from ._robot_collision import RobotCollisionSpherized
from ._geometry import CollGeom
import jaxlie
from typing import cast


@jdc.pytree_dataclass
class NeuralRobotCollisionSpherized(RobotCollisionSpherized):
    """
    A subclass of RobotCollisionSpherized that uses a neural network to approximate
    collision distances for a specific static environment (set of obstacles).
    
    The network is trained to overfit to a specific scene, mapping robot link poses
    directly to collision distances between robot links and the static obstacles.
    
    Input: Flattened link poses (N links × 7 pose params = N*7 dimensions)
    Output: Flattened distance matrix (N links × M obstacles = N*M dimensions)
    """
    
    # Neural network parameters (weights and biases for each layer)
    # We store them as a list of arrays.
    nn_params: List[Tuple[Float[Array, "fan_in fan_out"], Float[Array, "fan_out"]]] = jdc.field(default_factory=list)
    
    # Metadata about the training - these must be static for use in JIT conditionals
    is_trained: jdc.Static[bool] = False
    
    # We keep track of the number of obstacles this network was trained for (M)
    trained_num_obstacles: jdc.Static[int] = 0
    
    # Input normalization parameters (computed during training)
    input_mean: jax.Array = jdc.field(default_factory=lambda: jnp.zeros(1))
    input_std: jax.Array = jdc.field(default_factory=lambda: jnp.ones(1))

    @staticmethod
    def from_existing(
        original: RobotCollisionSpherized,
        layer_sizes: List[int] = None,
        key: jax.Array = None
    ) -> "NeuralRobotCollisionSpherized":
        """
        Creates a NeuralRobotCollisionSpherized instance from an existing RobotCollisionSpherized object.
        Initializes the neural network with random weights.
        
        Args:
            original: The original collision model.
            layer_sizes: List of hidden layer sizes. The input size is determined by robot DOF,
                         and output size by num_links * num_obstacles (determined at training time).
                         For initialization, we just set up the structure.
            key: JAX PRNG key for initialization.
        """
        if layer_sizes is None:
            layer_sizes = [256, 256, 256]
            
        if key is None:
            key = jax.random.PRNGKey(0)

        # We can't fully initialize the network structure until we know the output dimension (N*M),
        # which depends on the number of obstacles M. 
        # For now, we just copy the fields and return an untrained instance.
        # The actual weights will be initialized/shaped during the training setup or first call.
        
        return NeuralRobotCollisionSpherized(
            num_links=original.num_links,
            link_names=original.link_names,
            coll=original.coll,
            active_idx_i=original.active_idx_i,
            active_idx_j=original.active_idx_j,
            nn_params=[],
            is_trained=False,
            trained_num_obstacles=0
        )

    def _forward_nn(self, x: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        """
        Forward pass of the MLP.
        """
        # Simple MLP with ReLU activations
        for i, (w, b) in enumerate(self.nn_params):
            x = x @ w + b
            if i < len(self.nn_params) - 1:
                x = jax.nn.relu(x)
        return x

    @jdc.jit
    def at_config(
        self, robot: "Robot", cfg: Float[Array, "*batch actuated_count"]
    ) -> "CollGeom":
        """
        Returns the collision geometry transformed to the given robot configuration.

        This override fixes the shape mismatch in the parent class by extracting
        the transform for each specific link before applying it.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        assert self.link_names == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )
        
        Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg)
        
        coll_transformed = []
        for link in range(len(self.coll)):
            # Extract transform for this specific link: shape (*batch, 7)
            Ts_this_link = jaxlie.SE3(Ts_link_world_wxyz_xyz[..., link, :])
            coll_transformed.append(self.coll[link].transform(Ts_this_link))
        coll_transformed = cast(CollGeom, jax.tree.map(lambda *args: jnp.stack(args), *coll_transformed))
        
        return coll_transformed

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: "CollGeom",  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Overrides the compute_world_collision_distance to use the trained neural network.
        
        This assumes that world_geom represents the SAME static obstacles that the network
        was trained on. The network uses link poses (from forward kinematics) as input
        and predicts distances based on those poses.
        """
        if not self.is_trained:
            # Fallback to the original exact computation if not trained
            return super().compute_world_collision_distance(robot, cfg, world_geom)

        # Determine batch shapes
        batch_cfg_shape = cfg.shape[:-1]
        
        # Check world geom shape to ensure consistency with training (M)
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:
            M = 1
            batch_world_shape = ()
        else:
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]
            
        if M != self.trained_num_obstacles:
            logger.warning(
                f"Neural network was trained for {self.trained_num_obstacles} obstacles, "
                f"but current world_geom has {M}. Falling back to exact computation."
            )
            return super().compute_world_collision_distance(robot, cfg, world_geom)

        # Compute link poses via forward kinematics
        # Shape: (*batch_cfg, num_links, 7) where 7 = wxyz (4) + xyz (3)
        link_poses = robot.forward_kinematics(cfg)
        N = self.num_links
        
        # Flatten link poses to use as network input
        # Shape: (*batch_cfg, num_links * 7)
        link_poses_flat = link_poses.reshape(*batch_cfg_shape, N * 7)
        
        # Apply input normalization (using stored mean/std from training)
        link_poses_normalized = (link_poses_flat - self.input_mean) / self.input_std
        
        # Flatten batch for inference
        input_flat = link_poses_normalized.reshape(-1, N * 7)
        
        # Run inference
        predict_fn = jax.vmap(self._forward_nn)
        dists_flat = predict_fn(input_flat)  # Shape: (batch_size, N * M)
        
        # Reshape output to (*batch_cfg, N, M)
        dists = dists_flat.reshape(*batch_cfg_shape, N, M)
        
        # Handle broadcasting with world batch shape if necessary.
        if batch_world_shape:
             expected_batch_combined = jnp.broadcast_shapes(batch_cfg_shape, batch_world_shape)
             dists = jnp.broadcast_to(dists, (*expected_batch_combined, N, M))

        return dists

    def train(
        self,
        robot: "Robot",
        world_geom: "CollGeom",
        num_samples: int = 10000,
        batch_size: int = 256,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        key: jax.Array = None,
        layer_sizes: List[int] = [256, 256, 256]
    ) -> "NeuralRobotCollisionSpherized":
        """
        Trains the neural network to approximate the collision distances for the given world_geom.
        Returns a new instance with trained weights.

        The network maps from link poses (N*7 dimensions) to distances (N*M dimensions).
        Using full SE3 poses (quaternion + position) since link orientation affects
        where collision spheres end up in world space.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        key_samples, key_init, key_train = jax.random.split(key, 3)

        N = self.num_links
        world_axes = world_geom.get_batch_axes()
        M = world_axes[-1] if len(world_axes) > 0 else 1

        # 1. Generate training data
        logger.info(f"Generating {num_samples} samples for training...")

        # Sample random configurations within robot joint limits
        dof = robot.joints.num_actuated_joints
        lower_limits = robot.joints.lower_limits
        upper_limits = robot.joints.upper_limits
        q_train = jax.random.uniform(
            key_samples, (num_samples, dof), minval=lower_limits, maxval=upper_limits
        )

        # Compute link poses for all configurations via forward kinematics
        # Shape: (num_samples, num_links, 7) where 7 = wxyz (4) + xyz (3)
        logger.info("Computing link poses via forward kinematics...")
        link_poses_all = robot.forward_kinematics(q_train)
        
        # Flatten link poses to (num_samples, num_links * 7)
        X_train_raw = link_poses_all.reshape(num_samples, N * 7)
        
        # Normalize inputs: compute mean and std for better training
        X_mean = jnp.mean(X_train_raw, axis=0, keepdims=True)
        X_std = jnp.std(X_train_raw, axis=0, keepdims=True) + 1e-8
        X_train = (X_train_raw - X_mean) / X_std

        # 2. Compute ground truth labels (full N*M distances per sample)
        logger.info("Computing ground truth distances...")
        Y_train_rows = []
        for i in range(num_samples):
            q = q_train[i]
            # Exact distances for a single configuration: (N, M)
            dists = super().compute_world_collision_distance(robot, q, world_geom)
            # Flatten to (N*M,)
            dists_flat = dists.reshape(-1)
            Y_train_rows.append(dists_flat)

        # Final labels: (num_samples, N*M)
        Y_train = jnp.stack(Y_train_rows, axis=0)

        # 3. Initialize Network
        input_dim = N * 7  # num_links * 7 (wxyz_xyz pose representation)
        output_dim = N * M  # num_links * num_obstacles

        sizes = [input_dim] + layer_sizes + [output_dim]
        params = []
        k = key_init
        for i in range(len(sizes) - 1):
            k, subk = jax.random.split(k)
            fan_in, fan_out = sizes[i], sizes[i + 1]
            w = jax.random.normal(subk, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros((fan_out,))
            params.append((w, b))

        logger.info(
            f"Training neural network (Input: {input_dim} [link positions], Output: {output_dim} [distances])..."
        )

        # 4. Training Loop
        def loss_fn(p, x, y):
            curr_x = x
            for i, (w, b) in enumerate(p):
                curr_x = curr_x @ w + b
                if i < len(p) - 1:
                    curr_x = jax.nn.relu(curr_x)
            pred = curr_x  # (batch_size, output_dim)
            
            # Base MSE loss for distance regression
            mse_loss = jnp.mean((pred - y) ** 2)
            
            return mse_loss

        grad_fn = jax.value_and_grad(loss_fn)

        params_state = params

        # Adam state
        m = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        v = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0

        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            key_train, subk = jax.random.split(key_train)
            perm = jax.random.permutation(subk, num_samples)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            epoch_loss = 0.0

            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = start + batch_size
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]

                loss_val, grads = grad_fn(params_state, x_batch, y_batch)
                epoch_loss += loss_val

                # Adam update
                t += 1
                new_params = []
                new_m = []
                new_v = []

                for i in range(len(params_state)):
                    w, b = params_state[i]
                    dw, db = grads[i]
                    mw, mb = m[i]
                    vw, vb = v[i]

                    mw = beta1 * mw + (1.0 - beta1) * dw
                    mb = beta1 * mb + (1.0 - beta1) * db
                    vw = beta2 * vw + (1.0 - beta2) * (dw ** 2)
                    vb = beta2 * vb + (1.0 - beta2) * (db ** 2)

                    m_hat_w = mw / (1.0 - beta1 ** t)
                    m_hat_b = mb / (1.0 - beta1 ** t)
                    v_hat_w = vw / (1.0 - beta2 ** t)
                    v_hat_b = vb / (1.0 - beta2 ** t)

                    w_new = w - learning_rate * m_hat_w / (jnp.sqrt(v_hat_w) + epsilon)
                    b_new = b - learning_rate * m_hat_b / (jnp.sqrt(v_hat_b) + epsilon)

                    new_params.append((w_new, b_new))
                    new_m.append((mw, mb))
                    new_v.append((vw, vb))

                params_state = new_params
                m = new_m
                v = new_v

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Loss = {epoch_loss / num_batches:.6f}"
                )

        logger.info("Training complete.")

        return jdc.replace(
            self,
            nn_params=params_state,
            is_trained=True,
            trained_num_obstacles=M,
            input_mean=X_mean.squeeze(0),
            input_std=X_std.squeeze(0),
        )