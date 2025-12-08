import contextlib
import time
from functools import partial
from typing import Generator

import jax
import jax.numpy as jnp
import termcolor
from jaxtyping import Array, Float
from loguru import logger


# Icosahedron vertices for positional encoding directions (from iSDF)
# These 20 directions provide good coverage of the unit sphere
_ICOSAHEDRON_DIRS = jnp.array([
    [0.8506508, 0, 0.5257311],
    [0.809017, 0.5, 0.309017],
    [0.5257311, 0.8506508, 0],
    [1, 0, 0],
    [0.809017, 0.5, -0.309017],
    [0.8506508, 0, -0.5257311],
    [0.309017, 0.809017, -0.5],
    [0, 0.5257311, -0.8506508],
    [0.5, 0.309017, -0.809017],
    [0, 1, 0],
    [-0.5257311, 0.8506508, 0],
    [-0.309017, 0.809017, -0.5],
    [0, 0.5257311, 0.8506508],
    [-0.309017, 0.809017, 0.5],
    [0.309017, 0.809017, 0.5],
    [0.5, 0.309017, 0.809017],
    [0.5, -0.309017, 0.809017],
    [0, 0, 1],
    [-0.5, 0.309017, 0.809017],
    [-0.809017, 0.5, 0.309017],
]).T  # Shape: (3, 20) for efficient matmul


# First 50 prime numbers for Halton sequence bases
_HALTON_PRIMES = jnp.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
])


def positional_encoding(
    x: Float[Array, "... D"],
    min_deg: int = 0,
    max_deg: int = 6,
    scale: float = 1.0,
) -> Float[Array, "... embed_dim"]:
    """
    Standard sinusoidal positional encoding (NeRF-style).
    
    Applies sinusoidal encoding at multiple frequency bands to each input dimension.
    This helps the network capture high-frequency spatial details.
    
    Args:
        x: Input tensor of shape (..., D).
        min_deg: Minimum frequency degree (default 0).
        max_deg: Maximum frequency degree (default 6).
        scale: Scale factor applied to input before encoding (default 1.0).
    
    Returns:
        Positional encoding of shape (..., D + D * 2 * num_freqs).
    """
    n_freqs = max_deg - min_deg + 1
    # Frequency bands: 2^min_deg, 2^(min_deg+1), ..., 2^max_deg
    frequency_bands = 2.0 ** jnp.arange(min_deg, max_deg + 1)
    
    # Scale input
    x_scaled = x * scale
    
    # Start with original features
    embeddings = [x_scaled]
    
    # Apply sin and cos at each frequency to each dimension
    for freq in frequency_bands:
        embeddings.append(jnp.sin(freq * x_scaled))
        embeddings.append(jnp.cos(freq * x_scaled))
    
    # Concatenate all embeddings along last axis
    return jnp.concatenate(embeddings, axis=-1)


def compute_positional_encoding_dim(input_dim: int, min_deg: int = 0, max_deg: int = 6) -> int:
    """
    Compute the output dimension of positional encoding.
    
    Args:
        input_dim: Input dimension D.
        min_deg: Minimum frequency degree.
        max_deg: Maximum frequency degree.
    
    Returns:
        Output dimension after positional encoding.
    """
    n_freqs = max_deg - min_deg + 1
    # Original features + sin/cos for each frequency band applied to each dimension
    return input_dim + input_dim * 2 * n_freqs


def halton_sequence(num_samples: int, dim: int, skip: int = 100) -> jax.Array:
    """
    Generate a Halton sequence for quasi-random sampling.
    
    The Halton sequence provides better coverage of the sample space compared to
    uniform random sampling, which is beneficial for diverse sample collection.
    
    Args:
        num_samples: Number of samples to generate.
        dim: Dimensionality of each sample.
        skip: Number of initial samples to skip (improves uniformity).
    
    Returns:
        JAX array of shape (num_samples, dim) with values in [0, 1].
    """
    if dim > len(_HALTON_PRIMES):
        raise ValueError(f"Halton sequence dimension {dim} exceeds available primes ({len(_HALTON_PRIMES)})")
    
    # Generate samples using vectorized operations where possible
    indices = jnp.arange(skip, skip + num_samples)
    bases = _HALTON_PRIMES[:dim]
    
    # Compute maximum number of digits needed for the largest index
    max_index = skip + num_samples
    max_digits = int(jnp.ceil(jnp.log(max_index + 1) / jnp.log(2))) + 1
    
    def halton_for_base(base: int) -> jax.Array:
        """Vectorized Halton sequence for a single base."""
        # Compute radical inverse for all indices at once
        result = jnp.zeros(num_samples)
        f = 1.0 / base
        current = indices.astype(jnp.float32)
        
        for _ in range(max_digits):
            digit = jnp.mod(current, base)
            result = result + f * digit
            current = jnp.floor(current / base)
            f = f / base
        
        return result
    
    # Stack results for all dimensions
    samples = jnp.stack([halton_for_base(int(b)) for b in bases], axis=1)
    
    return samples


def rebalance_samples(
    samples: jax.Array,
    distances: jax.Array,
    num_samples: int,
    key: jax.Array,
    lower_limits: jax.Array,
    upper_limits: jax.Array,
    distance_fn: callable,
    collision_threshold: float = 0.1,
    target_collision_ratio: float = 0.8,
    target_near_ratio: float = 0.15,
    max_augment_iterations: int = 10,
    perturbation_scale_collision: float = 0.05,
    perturbation_scale_near: float = 0.03,
) -> jax.Array:
    """
    Rebalance a sample pool to have a target distribution of collision, near-collision,
    and free-space samples.
    
    Args:
        samples: Initial sample pool of shape (pool_size, dof).
        distances: Minimum distances for each sample, shape (pool_size,).
        num_samples: Target number of samples to return.
        key: JAX PRNG key.
        lower_limits: Lower joint limits, shape (dof,).
        upper_limits: Upper joint limits, shape (dof,).
        distance_fn: Function that computes minimum distance for a batch of samples.
                     Should have signature: distance_fn(samples) -> distances.
        collision_threshold: Distance threshold for near-collision (default 0.1).
        target_collision_ratio: Target ratio of collision samples (default 0.8).
        target_near_ratio: Target ratio of near-collision samples (default 0.15).
        max_augment_iterations: Maximum iterations for augmentation (default 10).
        perturbation_scale_collision: Perturbation scale for collision augmentation (default 0.05).
        perturbation_scale_near: Perturbation scale for near-collision augmentation (default 0.03).
    
    Returns:
        Rebalanced samples of shape (num_samples, dof).
    """
    dof = samples.shape[1]
    
    # Separate samples into categories
    is_in_collision = distances <= 0
    is_near_collision = (distances > 0) & (distances < collision_threshold)
    is_free_space = distances >= collision_threshold
    
    collision_samples = samples[is_in_collision]
    near_collision_samples = samples[is_near_collision]
    free_space_samples = samples[is_free_space]
    
    num_collision = collision_samples.shape[0]
    num_near_collision = near_collision_samples.shape[0]
    num_free = free_space_samples.shape[0]
    
    logger.info(f"Sample distribution from pool: collision={num_collision}, near-collision={num_near_collision}, free-space={num_free}")
    
    # Target distribution
    target_collision = int(num_samples * target_collision_ratio)
    target_near = int(num_samples * target_near_ratio)
    target_free = num_samples - target_collision - target_near
    
    key_augment = key
    
    # Augment collision samples if needed
    if num_collision < target_collision and num_collision > 0:
        logger.info(f"Augmenting collision samples from {num_collision} to {target_collision}...")
        
        samples_needed = target_collision - num_collision
        augmented_list = []
        iteration = 0
        
        while len(augmented_list) < samples_needed and iteration < max_augment_iterations:
            iteration += 1
            batch_size_aug = min(samples_needed * 2, 5000)
            key_augment, subk1, subk2 = jax.random.split(key_augment, 3)
            indices = jax.random.randint(subk1, (batch_size_aug,), 0, num_collision)
            base_samples = collision_samples[indices]
            
            perturbation_range = perturbation_scale_collision * (upper_limits - lower_limits)
            perturbations = jax.random.uniform(subk2, (batch_size_aug, dof), minval=-1, maxval=1) * perturbation_range
            candidates = jnp.clip(base_samples + perturbations, lower_limits, upper_limits)
            
            candidate_dists = distance_fn(candidates)
            valid_mask = candidate_dists <= 0
            valid_candidates = candidates[valid_mask]
            
            if valid_candidates.shape[0] > 0:
                augmented_list.append(valid_candidates)
                
            logger.debug(f"  Iteration {iteration}: {valid_candidates.shape[0]} valid collision samples generated")
        
        if augmented_list:
            all_augmented = jnp.concatenate(augmented_list, axis=0)[:samples_needed]
            collision_samples = jnp.concatenate([collision_samples, all_augmented], axis=0)
            num_collision = collision_samples.shape[0]
            logger.info(f"  Final collision sample count: {num_collision}")
    
    # Augment near-collision samples if needed
    if num_near_collision < target_near and num_near_collision > 0:
        logger.info(f"Augmenting near-collision samples from {num_near_collision} to {target_near}...")
        
        samples_needed = target_near - num_near_collision
        augmented_list = []
        iteration = 0
        
        while len(augmented_list) < samples_needed and iteration < max_augment_iterations:
            iteration += 1
            batch_size_aug = min(samples_needed * 2, 5000)
            key_augment, subk1, subk2 = jax.random.split(key_augment, 3)
            indices = jax.random.randint(subk1, (batch_size_aug,), 0, num_near_collision)
            base_samples = near_collision_samples[indices]
            
            perturbation_range = perturbation_scale_near * (upper_limits - lower_limits)
            perturbations = jax.random.uniform(subk2, (batch_size_aug, dof), minval=-1, maxval=1) * perturbation_range
            candidates = jnp.clip(base_samples + perturbations, lower_limits, upper_limits)
            
            candidate_dists = distance_fn(candidates)
            valid_mask = (candidate_dists > 0) & (candidate_dists < collision_threshold)
            valid_candidates = candidates[valid_mask]
            
            if valid_candidates.shape[0] > 0:
                augmented_list.append(valid_candidates)
        
        if augmented_list:
            all_augmented = jnp.concatenate(augmented_list, axis=0)[:samples_needed]
            near_collision_samples = jnp.concatenate([near_collision_samples, all_augmented], axis=0)
            num_near_collision = near_collision_samples.shape[0]
            logger.info(f"  Final near-collision sample count: {num_near_collision}")
    
    # Construct final training set
    actual_collision = min(num_collision, target_collision)
    actual_near = min(num_near_collision, target_near)
    actual_free = max(0, num_samples - actual_collision - actual_near)
    actual_free = min(actual_free, num_free)
    
    logger.info(f"Assembling training set: collision={actual_collision}, near={actual_near}, free={actual_free}")
    
    # Select samples from each category
    key_augment, subk = jax.random.split(key_augment)
    
    selected_collision = collision_samples[:actual_collision] if actual_collision > 0 else jnp.empty((0, dof))
    selected_near = near_collision_samples[:actual_near] if actual_near > 0 else jnp.empty((0, dof))
    
    if actual_free > 0 and num_free > 0:
        free_indices = jax.random.choice(subk, num_free, shape=(actual_free,), replace=False)
        selected_free = free_space_samples[free_indices]
    else:
        selected_free = jnp.empty((0, dof))
    
    # Combine all samples
    parts = [p for p in [selected_collision, selected_near, selected_free] if p.shape[0] > 0]
    result = jnp.concatenate(parts, axis=0) if parts else jnp.empty((0, dof))
    
    # Fill shortfall from original pool if needed
    if result.shape[0] < num_samples:
        shortfall = num_samples - result.shape[0]
        logger.info(f"Filling shortfall of {shortfall} samples from original pool...")
        key_augment, subk = jax.random.split(key_augment)
        extra_indices = jax.random.choice(subk, samples.shape[0], shape=(shortfall,), replace=True)
        extra_samples = samples[extra_indices]
        result = jnp.concatenate([result, extra_samples], axis=0)
    
    # Shuffle the result
    key_augment, subk = jax.random.split(key_augment)
    shuffle_perm = jax.random.permutation(subk, result.shape[0])
    result = result[shuffle_perm][:num_samples]
    
    logger.info(f"Final training set: {result.shape[0]} samples")
    
    return result


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


def _log(fmt: str, *args, **kwargs) -> None:
    logger.bind(function="log").info(fmt, *args, **kwargs)


def jax_log(fmt: str, *args, **kwargs) -> None:
    """Emit a loguru info message from a JITed JAX function."""
    jax.debug.callback(partial(_log, fmt), *args, **kwargs)

@partial(jax.jit, static_argnames=['dtype'])
def quantize(tree, dtype=jax.numpy.float16):
    return jax.tree.map(lambda x: x.astype(dtype) if hasattr(x, 'astype') else x, tree)