"""Quantize Collision Example

Script to generate obstacles and robot configurations for quantization testing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import pyronot as pk
from pyronot.collision import RobotCollision, RobotCollisionSpherized, NeuralRobotCollision, Sphere
from pyronot._robot_urdf_parser import RobotURDFParser
import yourdfpy
from pyronot.utils import quantize

# Global configuration: Set to True to run positional encoding comparison
RUN_POSITIONAL_ENCODING = False

def generate_spheres(n_spheres):
    print(f"Generating {n_spheres} random spheres...")
    spheres = []
    for _ in range(n_spheres):
        center = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        radius = np.random.uniform(low=0.05, high=0.2)
        sphere = Sphere.from_center_and_radius(center, np.array(radius))
        spheres.append(sphere)
    
    # Tree map them to create a batch of spheres
    spheres_batch = jax.tree.map(lambda *args: jnp.stack(args), *spheres)
    print(f"Generated {n_spheres} spheres.")
    return spheres_batch

def generate_configs(joints, n_configs):
    print(f"Generating {n_configs} random robot configurations...")
    q_batch = np.random.uniform(
        low=joints.lower_limits, 
        high=joints.upper_limits, 
        size=(n_configs, joints.num_actuated_joints)
    )
    print(f"Generated {n_configs} robot configurations.")
    print(f"Configurations shape: {q_batch.shape}")
    return q_batch

def make_collision_checker(robot, robot_coll):
    @jax.jit
    def check_collisions(q_batch, obstacles):
        # q_batch: (N_configs, dof)
        # obstacles: Sphere batch (N_spheres)
        
        # Define single config check
        def check_single(q, obs):
            return robot_coll.compute_world_collision_distance(robot, q, obs)
            
        # Vmap over configs
        # in_axes: q=(0), obs=(None) -> we want to check each q against ALL obs
        return jax.vmap(check_single, in_axes=(0, None))(q_batch, obstacles)
    
    return check_collisions


def make_neural_collision_checker(robot, robot_coll, spheres_batch, use_positional_encoding=True, pe_min_deg=0, pe_max_deg=6):
    """Train a neural collision model on the given static world and return its checker.

    This will:
    - build a NeuralRobotCollision from the exact model
    - train it on random configs for the provided world (spheres_batch)
    - expose a vmap'ed collision function with the same signature as make_collision_checker's output
    
    Args:
        robot: The Robot instance
        robot_coll: The exact collision model (RobotCollisionSpherized)
        spheres_batch: Batch of sphere obstacles
        use_positional_encoding: If True, use iSDF-inspired positional encoding for
                                  better capture of fine geometric details (default True)
        pe_min_deg: Minimum frequency degree for positional encoding (default 0)
        pe_max_deg: Maximum frequency degree for positional encoding (default 6)
    """

    # Wrap the world geometry in the same structure RobotCollisionSpherized expects
    # RobotCollisionSpherized.from_urdf constructs a CollGeom internally when used in examples,
    # so `spheres_batch` is already a valid batch of Sphere geometry.

    # Create neural collision model from existing exact model
    # With positional encoding enabled for better accuracy near collision boundaries
    neural_coll = NeuralRobotCollision.from_existing(
        robot_coll,
        use_positional_encoding=use_positional_encoding,
        pe_min_deg=pe_min_deg,
        pe_max_deg=pe_max_deg,
    )
    
    pe_status = f"with PE (deg {pe_min_deg}-{pe_max_deg})" if use_positional_encoding else "without PE"
    print(f"Created neural collision model {pe_status}")

    # Train neural model on this specific world
    neural_coll = neural_coll.train(
        robot=robot,
        world_geom=spheres_batch,
        num_samples=10000,
        batch_size=1000,  # Smaller batch = more gradient updates per epoch
        epochs=50,      # More epochs for better convergence
        learning_rate=1e-3,
    )

    # Now build a collision checker that calls the neural model
    @jax.jit
    def check_collisions(q_batch, obstacles):
        # q_batch: (N_configs, dof)
        # obstacles: same spheres_batch used for training

        def check_single(q, obs):
            return neural_coll.compute_world_collision_distance(robot, q, obs)

        return jax.vmap(check_single, in_axes=(0, None))(q_batch, obstacles)

    return check_collisions

def run_benchmark(name, check_fn, q_batch, obstacles):
    print(f"\n{name}:")
    
    # Metrics
    q_size_mb = q_batch.nbytes / 1024 / 1024
    spheres_size_mb = sum(x.nbytes for x in jax.tree_util.tree_leaves(obstacles)) / 1024 / 1024
    
    print(f"q_batch size: {q_size_mb:.2f} MB")
    print(f"Obstacles (spheres) size: {spheres_size_mb:.2f} MB")

    # Warmup
    print(f"Warming up JIT ({name})...")
    _ = check_fn(q_batch, obstacles)

    # Run collision checking
    print(f"Executing collision checking ({name})...")
    start_time = time.perf_counter()
    dists = check_fn(q_batch, obstacles)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Time to compute: {elapsed_time:.6f} seconds")
    print(f"Collision distances shape: {dists.shape}")
    print(f"Min distance: {jnp.min(dists):.6f}")
    print(f"Max distance: {jnp.max(dists):.6f}")
    print(f"Mean distance: {jnp.mean(dists):.6f}")
    print(f"Std distance: {jnp.std(dists):.6f}")
    
    in_collision = dists < 0
    print(f"Number of collision pairs: {jnp.sum(in_collision)}")
    
    return dists, elapsed_time


def compare_results(name, neural_dists, exact_dists):
    """Compare neural network predictions against exact results."""
    import gc
    
    print(f"\n {name} Comparison ")

    
    # Compute metrics in a memory-efficient way
    diff = neural_dists - exact_dists
    mae = float(jnp.mean(jnp.abs(diff)))
    max_ae = float(jnp.max(jnp.abs(diff)))
    rmse = float(jnp.sqrt(jnp.mean(diff ** 2)))
    bias = float(jnp.mean(diff))
    del diff  # Free memory
    gc.collect()
    
    print(f"Mean absolute error: {mae:.6f}")
    print(f"Max absolute error: {max_ae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean error (bias): {bias:.6f}")
    
    # Check accuracy at collision boundary
    exact_in_collision = exact_dists < 0.05
    neural_in_collision = neural_dists < 0.05
    
    # Compute metrics and convert to Python ints immediately
    true_positives = int(jnp.sum(exact_in_collision & neural_in_collision))
    false_positives = int(jnp.sum(~exact_in_collision & neural_in_collision))
    false_negatives = int(jnp.sum(exact_in_collision & ~neural_in_collision))
    true_negatives = int(jnp.sum(~exact_in_collision & ~neural_in_collision))
    
    # Free the boolean arrays
    del exact_in_collision, neural_in_collision
    gc.collect()
    
    print(f"\nCollision Detection Accuracy (threshold=0.05):")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  True Negatives: {true_negatives}")
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {
        'mae': mae,
        'max_ae': max_ae,
        'rmse': rmse,
        'bias': bias,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    import gc
    
    # Load robot
    urdf_path = "resources/ur5/ur5_spherized.urdf"
    mesh_dir = "resources/ur5/meshes"
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    joints, links = RobotURDFParser.parse(urdf)
    
    # Initialize collision model
    print("Initializing collision model...")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    # Generate data (world is fixed for both exact and neural models)
    spheres_batch = generate_spheres(100)
    q_batch = generate_configs(joints, 50000)

    # Create collision checker using exact model
    exact_check_collisions = make_collision_checker(robot, robot_coll)

    # Train neural model without positional encoding (default)
    print("Training neural collision model WITHOUT positional encoding...")
    print("(Raw link poses as input)")
    neural_check_without_pe = make_neural_collision_checker(
        robot, robot_coll, spheres_batch,
        use_positional_encoding=False,
    )

    # Optionally train with positional encoding
    neural_check_with_pe = None
    if RUN_POSITIONAL_ENCODING:
        print("\n" + "="*70)
        print("Training neural collision model WITH positional encoding...")
        print("(iSDF-inspired: projects onto icosahedron directions with frequency bands)")
        print("="*70)
        neural_check_with_pe = make_neural_collision_checker(
            robot, robot_coll, spheres_batch,
            use_positional_encoding=True,
            pe_min_deg=0,
            pe_max_deg=6,  # 7 frequency bands: 2^0, 2^1, ..., 2^6
        )

    # Run benchmarks
    print("\n" + "="*70)
    print("Running benchmarks...")
    print("="*70)
    
    exact_dists, exact_time = run_benchmark(
        "Exact (RobotCollisionSpherized)", 
        exact_check_collisions, q_batch, spheres_batch
    )
    
    neural_without_pe_dists, neural_without_pe_time = run_benchmark(
        "Neural WITHOUT Positional Encoding", 
        neural_check_without_pe, q_batch, spheres_batch
    )
    
    neural_with_pe_dists = None
    neural_with_pe_time = None
    if RUN_POSITIONAL_ENCODING and neural_check_with_pe is not None:
        neural_with_pe_dists, neural_with_pe_time = run_benchmark(
            "Neural WITH Positional Encoding", 
            neural_check_with_pe, q_batch, spheres_batch
        )
    
    # Clear JAX caches and force garbage collection to free GPU memory
    print("\nClearing memory before comparison...")
    jax.clear_caches()
    gc.collect()
    
    # Compare results
    metrics_without_pe = compare_results(
        "Neural WITHOUT Positional Encoding vs Exact",
        neural_without_pe_dists, exact_dists
    )
    
    metrics_with_pe = None
    if RUN_POSITIONAL_ENCODING and neural_with_pe_dists is not None:
        metrics_with_pe = compare_results(
            "Neural WITH Positional Encoding vs Exact",
            neural_with_pe_dists, exact_dists
        )
    
    # Summary comparison (only if positional encoding was tested)
    if RUN_POSITIONAL_ENCODING and metrics_with_pe is not None:
        print("SUMMARY: Positional Encoding Impact")
        print(f"\n{'Metric':<25} {'With PE':<15} {'Without PE':<15} {'Improvement':<15}")
        
        for metric in ['mae', 'rmse', 'max_ae', 'precision', 'recall', 'f1']:
            with_pe = metrics_with_pe[metric]
            without_pe = metrics_without_pe[metric]
            
            # For error metrics, lower is better; for accuracy metrics, higher is better
            if metric in ['mae', 'rmse', 'max_ae', 'bias']:
                improvement = (without_pe - with_pe) / (without_pe + 1e-8) * 100
                better = "↓" if with_pe < without_pe else "↑"
            else:
                improvement = (with_pe - without_pe) / (without_pe + 1e-8) * 100
                better = "↑" if with_pe > without_pe else "↓"
            
            print(f"{metric:<25} {with_pe:<15.6f} {without_pe:<15.6f} {improvement:+.1f}% {better}")
        
        print(f"\n{'Inference Time (s)':<25} {neural_with_pe_time:<15.6f} {neural_without_pe_time:<15.6f}")
        print(f"{'Exact Time (s)':<25} {exact_time:<15.6f}")
    else:
        # Simple summary without PE comparison
        print("SUMMARY: Neural vs Exact")
        print(f"\n{'Metric':<25} {'Neural (no PE)':<15}")
        for metric in ['mae', 'rmse', 'max_ae', 'precision', 'recall', 'f1']:
            print(f"{metric:<25} {metrics_without_pe[metric]:<15.6f}")
        print(f"\n{'Neural Inference Time (s)':<25} {neural_without_pe_time:<15.6f}")
        print(f"{'Exact Time (s)':<25} {exact_time:<15.6f}")
        print(f"{'Speedup':<25} {exact_time / neural_without_pe_time:<15.2f}x")
    
    # Cleanup
    del exact_dists, neural_without_pe_dists
    if neural_with_pe_dists is not None:
        del neural_with_pe_dists
    gc.collect()


if __name__ == "__main__":
    main()
