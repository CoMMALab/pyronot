from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float
from loguru import logger

if TYPE_CHECKING:
    from pyronot._robot import Robot
    from ._geometry import CollGeom
from ._robot_collision import RobotCollisionSpherized, RobotCollision
from ._geometry import CollGeom
from pyronot.utils import (
    positional_encoding,
    compute_positional_encoding_dim,
    halton_sequence,
    rebalance_samples,
)
import jaxlie
from typing import cast


@jdc.pytree_dataclass
class NeuralRobotCollision:
    """
    A wrapper class that adds neural network-based collision distance approximation
    to either RobotCollision or RobotCollisionSpherized.
    
    The network is trained to overfit to a specific scene, mapping robot link poses
    directly to collision distances between robot links and the static obstacles.
    
    Input: Flattened link poses (N links × 7 pose params = N*7 dimensions), optionally
           with positional encoding for capturing fine geometric details.
    Output: Flattened distance matrix (N links × M obstacles = N*M dimensions)
    
    This class uses composition to wrap either collision model type, delegating
    non-neural methods to the underlying collision model.
    
    Positional Encoding (inspired by iSDF):
        When enabled, the input is augmented with sinusoidal positional encodings
        at multiple frequency scales. This allows the network to learn high-frequency
        spatial features that are critical for accurate collision distance prediction,
        especially near obstacle boundaries where distances change rapidly.
    """
    
    # The underlying collision model (either RobotCollision or RobotCollisionSpherized)
    _collision_model: Union[RobotCollision, RobotCollisionSpherized]
    
    # Neural network parameters (weights and biases for each layer)
    nn_params: List[Tuple[Float[Array, "fan_in fan_out"], Float[Array, "fan_out"]]] = jdc.field(default_factory=list)
    
    # Metadata about the training - these must be static for use in JIT conditionals
    is_trained: jdc.Static[bool] = False
    
    # We keep track of the number of obstacles this network was trained for (M)
    trained_num_obstacles: jdc.Static[int] = 0
    
    # Input normalization parameters (computed during training)
    input_mean: jax.Array = jdc.field(default_factory=lambda: jnp.zeros(1))
    input_std: jax.Array = jdc.field(default_factory=lambda: jnp.ones(1))
    
    # Positional encoding parameters (static for JIT)
    use_positional_encoding: jdc.Static[bool] = False
    pe_min_deg: jdc.Static[int] = 0
    pe_max_deg: jdc.Static[int] = 6
    pe_scale: jdc.Static[float] = 1.0
    
    # Computed PE scale (stored after training for use at inference)
    pe_scale_computed: jax.Array = jdc.field(default_factory=lambda: jnp.array(1.0))

    # Properties to expose underlying collision model attributes
    @property
    def num_links(self) -> int:
        return self._collision_model.num_links
    
    @property
    def link_names(self) -> tuple[str, ...]:
        return self._collision_model.link_names
    
    @property
    def coll(self) -> CollGeom:
        return self._collision_model.coll
    
    @property
    def active_idx_i(self):
        return self._collision_model.active_idx_i
    
    @property
    def active_idx_j(self):
        return self._collision_model.active_idx_j
    
    @property
    def is_spherized(self) -> bool:
        """Returns True if the underlying model is RobotCollisionSpherized."""
        return isinstance(self._collision_model, RobotCollisionSpherized)

    @staticmethod
    def from_existing(
        original: Union[RobotCollision, RobotCollisionSpherized],
        layer_sizes: List[int] = None,
        key: jax.Array = None,
        use_positional_encoding: bool = True,
        pe_min_deg: int = 0,
        pe_max_deg: int = 6,
        pe_scale: float = 1.0,
    ) -> "NeuralRobotCollision":
        """
        Creates a NeuralRobotCollision instance from an existing collision model.
        Initializes the neural network with random weights.
        
        Args:
            original: The original collision model (RobotCollision or RobotCollisionSpherized).
            layer_sizes: List of hidden layer sizes. The input size is determined by robot DOF,
                         and output size by num_links * num_obstacles (determined at training time).
                         For initialization, we just set up the structure.
            key: JAX PRNG key for initialization.
            use_positional_encoding: If True, use positional encoding on input (default True).
                                     Inspired by iSDF, this helps capture fine geometric details.
            pe_min_deg: Minimum frequency degree for positional encoding (default 0).
            pe_max_deg: Maximum frequency degree for positional encoding (default 6).
            pe_scale: Scale factor for positional encoding input (default 1.0).
        """
        if layer_sizes is None:
            layer_sizes = [256, 256, 256]
            
        if key is None:
            key = jax.random.PRNGKey(0)

        # We can't fully initialize the network structure until we know the output dimension (N*M),
        # which depends on the number of obstacles M. 
        # For now, we just copy the fields and return an untrained instance.
        # The actual weights will be initialized/shaped during the training setup or first call.
        
        return NeuralRobotCollision(
            _collision_model=original,
            nn_params=[],
            is_trained=False,
            trained_num_obstacles=0,
            use_positional_encoding=use_positional_encoding,
            pe_min_deg=pe_min_deg,
            pe_max_deg=pe_max_deg,
            pe_scale=pe_scale,
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
        Delegates to the underlying collision model.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        return self._collision_model.at_config(robot, cfg)

    def compute_self_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.
        Delegates to the underlying collision model.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """
        return self._collision_model.compute_self_collision_distance(robot, cfg)

    def get_swept_capsules(
        self,
        robot: "Robot",
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ):
        """
        Computes swept-volume capsules between two configurations.
        Delegates to the underlying collision model.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
        """
        return self._collision_model.get_swept_capsules(robot, cfg_prev, cfg_next)

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: "CollGeom",  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes collision distances, using the trained neural network if available.
        
        This assumes that world_geom represents the SAME static obstacles that the network
        was trained on. The network uses link poses (from forward kinematics) as input
        and predicts distances based on those poses.
        
        If positional encoding is enabled, the input is augmented with sinusoidal
        embeddings at multiple frequency scales to capture fine geometric details.
        
        Falls back to the underlying collision model's exact computation if not trained.
        """
        if not self.is_trained:
            # Fallback to the original exact computation if not trained
            return self._collision_model.compute_world_collision_distance(robot, cfg, world_geom)

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
            return self._collision_model.compute_world_collision_distance(robot, cfg, world_geom)

        # Compute link poses via forward kinematics
        # Shape: (*batch_cfg, num_links, 7) where 7 = wxyz (4) + xyz (3)
        link_poses = robot.forward_kinematics(cfg)
        N = self.num_links
        
        # Flatten link poses to use as network input
        # Shape: (*batch_cfg, num_links * 7)
        link_poses_flat = link_poses.reshape(*batch_cfg_shape, N * 7)
        
        # Apply positional encoding BEFORE normalization if enabled
        # (matching the training procedure)
        if self.use_positional_encoding:
            link_poses_pe = positional_encoding(
                link_poses_flat,
                min_deg=self.pe_min_deg,
                max_deg=self.pe_max_deg,
                scale=self.pe_scale_computed,  # Use the computed scale from training
            )
            # Then normalize
            link_poses_normalized = (link_poses_pe - self.input_mean) / self.input_std
        else:
            # Just normalize
            link_poses_normalized = (link_poses_flat - self.input_mean) / self.input_std
        
        # Flatten batch for inference
        input_dim = link_poses_normalized.shape[-1]
        input_flat = link_poses_normalized.reshape(-1, input_dim)
        
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
        batch_size: int = 1000,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        key: jax.Array = None,
        layer_sizes: List[int] = [256, 256, 256, 256]
    ) -> "NeuralRobotCollision":
        """
        Trains the neural network to approximate the collision distances for the given world_geom.
        Returns a new instance with trained weights.

        The network maps from link poses (N*7 dimensions) to distances (N*M dimensions).
        Using full SE3 poses (quaternion + position) since link orientation affects
        where collision spheres end up in world space.
        """
        logger.info("Starting neural collision training...")
        
        if key is None:
            key = jax.random.PRNGKey(0)

        key_samples, key_init, key_train = jax.random.split(key, 3)

        N = self.num_links
        world_axes = world_geom.get_batch_axes()
        M = world_axes[-1] if len(world_axes) > 0 else 1

        # 1. Generate training data with collision-aware sampling
        logger.info(f"Generating {num_samples} samples with collision-aware sampling...")

        # Sample configurations using Halton sequence for better space coverage
        dof = robot.joints.num_actuated_joints
        lower_limits = robot.joints.lower_limits
        upper_limits = robot.joints.upper_limits
        
        # Generate initial Halton sequence samples (2x to have pool for filtering)
        initial_pool_size = num_samples * 2
        halton_samples = halton_sequence(initial_pool_size, dof)
        q_pool = lower_limits + halton_samples * (upper_limits - lower_limits)
        
        # Compute distances for the pool to identify collision samples
        logger.info("Computing distances to identify collision samples...")
        
        def compute_min_dist(q):
            dists = self._collision_model.compute_world_collision_distance(
                robot, q, world_geom
            )
            return jnp.min(dists)
        
        compute_all_min_dists = jax.vmap(compute_min_dist)
        min_dists = compute_all_min_dists(q_pool)  # Shape: (initial_pool_size,)
        
        # Rebalance samples to have more collision and near-collision samples
        collision_threshold = 0.1  # Samples within 10cm of collision
        q_train = rebalance_samples(
            samples=q_pool,
            distances=min_dists,
            num_samples=num_samples,
            key=key_samples,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            distance_fn=compute_all_min_dists,
            collision_threshold=collision_threshold,
        )

        # Compute link poses for all configurations via forward kinematics
        # Shape: (num_samples, num_links, 7) where 7 = wxyz (4) + xyz (3)
        logger.info("Computing link poses via forward kinematics...")
        link_poses_all = robot.forward_kinematics(q_train)
        
        # Flatten link poses to (num_samples, num_links * 7)
        X_train_raw = link_poses_all.reshape(num_samples, N * 7)
        
        # Apply positional encoding BEFORE normalization if enabled
        # This is important because positional encoding should operate on the
        # original spatial scale of the data, not normalized values
        if self.use_positional_encoding:
            # For positional encoding, we want the input scaled so that the 
            # lowest frequency (2^min_deg) captures the full data range,
            # and higher frequencies capture finer details.
            # A scale of 1.0 is typically fine when data is already in reasonable ranges.
            # We use pi as scale so that the range [-1, 1] maps to [-pi, pi]
            auto_scale = jnp.pi
            logger.info(f"Applying positional encoding (min_deg={self.pe_min_deg}, max_deg={self.pe_max_deg}, scale={auto_scale:.4f})...")
            X_train_pe = positional_encoding(
                X_train_raw,
                min_deg=self.pe_min_deg,
                max_deg=self.pe_max_deg,
                scale=auto_scale,
            )
            logger.info(f"Input dimension after positional encoding: {X_train_pe.shape[-1]} (was {X_train_raw.shape[-1]})")
            
            # Now normalize the positional-encoded features
            X_mean = jnp.mean(X_train_pe, axis=0, keepdims=True)
            X_std = jnp.std(X_train_pe, axis=0, keepdims=True) + 1e-8
            X_train = (X_train_pe - X_mean) / X_std
            
            # Store the auto-computed scale for inference
            self_pe_scale_computed = auto_scale
        else:
            # Normalize inputs: compute mean and std for better training
            X_mean = jnp.mean(X_train_raw, axis=0, keepdims=True)
            X_std = jnp.std(X_train_raw, axis=0, keepdims=True) + 1e-8
            X_train = (X_train_raw - X_mean) / X_std
            self_pe_scale_computed = self.pe_scale

        # 2. Compute ground truth labels using vmap for acceleration
        logger.info("Computing ground truth distances (vectorized)...")
        
        # Use vmap to compute distances for all configurations in parallel
        def compute_single_dist(q):
            dists = self._collision_model.compute_world_collision_distance(
                robot, q, world_geom
            )
            return dists.reshape(-1)  # Flatten to (N*M,)
        
        # Vectorize over all training samples
        compute_all_dists = jax.vmap(compute_single_dist)
        Y_train = compute_all_dists(q_train)  # Shape: (num_samples, N*M)
        
        # Compute sample weights based on minimum distance
        # Give higher weight to collision and near-collision samples
        Y_min_per_sample = jnp.min(Y_train, axis=1)  # Shape: (num_samples,)
        
        # Weight function: higher weight for collision (dist <= 0) and near-collision
        # collision: weight = 3.0, near-collision: weight = 2.0, free: weight = 1.0
        sample_weights = jnp.where(
            Y_min_per_sample <= 0,
            3.0,  # Collision samples get 3x weight
            jnp.where(
                Y_min_per_sample < collision_threshold,
                2.0,  # Near-collision samples get 2x weight
                1.0   # Free space samples get normal weight
            )
        )
        # Normalize weights so they sum to num_samples (to maintain loss scale)
        sample_weights = sample_weights * (num_samples / jnp.sum(sample_weights))
        
        logger.info(f"Sample weights - collision (3x): {jnp.sum(Y_min_per_sample <= 0)}, near-collision (2x): {jnp.sum((Y_min_per_sample > 0) & (Y_min_per_sample < collision_threshold))}")

        # 3. Initialize Network
        # Input dimension depends on whether positional encoding is enabled
        raw_input_dim = N * 7  # num_links * 7 (wxyz_xyz pose representation)
        if self.use_positional_encoding:
            input_dim = compute_positional_encoding_dim(raw_input_dim, self.pe_min_deg, self.pe_max_deg)
        else:
            input_dim = raw_input_dim
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

        pe_info = f" with positional encoding (deg {self.pe_min_deg}-{self.pe_max_deg})" if self.use_positional_encoding else ""
        logger.info(
            f"Training neural network{pe_info} (Input: {input_dim}, Output: {output_dim} [distances])..."
        )

        # 4. Define JIT-compiled training step
        @jax.jit
        def forward_pass(params, x):
            """Forward pass through the network."""
            for i, (w, b) in enumerate(params):
                x = x @ w + b
                if i < len(params) - 1:
                    x = jax.nn.relu(x)
            return x

        @jax.jit
        def loss_fn(params, x, y, weights):
            """Compute weighted MSE loss."""
            pred = forward_pass(params, x)
            # Per-sample MSE, then weight and average
            sample_mse = jnp.mean((pred - y) ** 2, axis=1)  # (batch_size,)
            return jnp.mean(sample_mse * weights)

        @jax.jit
        def train_step(params, opt_state, x_batch, y_batch, w_batch, t):
            """Single training step with Adam optimizer."""
            m, v = opt_state
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            
            # Compute gradients
            loss_val, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch, w_batch)
            
            # Adam update
            new_params = []
            new_m = []
            new_v = []
            
            for i in range(len(params)):
                w, b = params[i]
                dw, db = grads[i]
                mw, mb = m[i]
                vw, vb = v[i]
                
                # Update biased first moment estimate
                mw = beta1 * mw + (1.0 - beta1) * dw
                mb = beta1 * mb + (1.0 - beta1) * db
                
                # Update biased second moment estimate
                vw = beta2 * vw + (1.0 - beta2) * (dw ** 2)
                vb = beta2 * vb + (1.0 - beta2) * (db ** 2)
                
                # Bias correction
                m_hat_w = mw / (1.0 - beta1 ** t)
                m_hat_b = mb / (1.0 - beta1 ** t)
                v_hat_w = vw / (1.0 - beta2 ** t)
                v_hat_b = vb / (1.0 - beta2 ** t)
                
                # Update parameters
                w_new = w - learning_rate * m_hat_w / (jnp.sqrt(v_hat_w) + epsilon)
                b_new = b - learning_rate * m_hat_b / (jnp.sqrt(v_hat_b) + epsilon)
                
                new_params.append((w_new, b_new))
                new_m.append((mw, mb))
                new_v.append((vw, vb))
            
            return new_params, (new_m, new_v), loss_val

        # Initialize Adam state
        m = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        v = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        opt_state = (m, v)
        
        params_state = params
        t = 0
        num_batches = num_samples // batch_size

        # 5. Training loop
        for epoch in range(epochs):
            key_train, subk = jax.random.split(key_train)
            perm = jax.random.permutation(subk, num_samples)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]
            W_shuffled = sample_weights[perm]  # Shuffle weights along with data

            epoch_loss = 0.0

            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = start + batch_size
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]
                w_batch = W_shuffled[start:end]

                t += 1
                params_state, opt_state, loss_val = train_step(
                    params_state, opt_state, x_batch, y_batch, w_batch, t
                )
                epoch_loss += loss_val

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
            pe_scale_computed=jnp.array(self_pe_scale_computed),
        )


# Backward compatibility alias
NeuralRobotCollisionSpherized = NeuralRobotCollision