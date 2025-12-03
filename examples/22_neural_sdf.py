"""Quantize Collision Example

Script to generate obstacles and robot configurations for quantization testing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import pyroki as pk
from pyroki.collision import RobotCollisionSpherized, NeuralRobotCollisionSpherized, Sphere
from pyroki._robot_urdf_parser import RobotURDFParser
import yourdfpy
from pyroki.utils import quantize

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


def make_neural_collision_checker(robot, robot_coll, spheres_batch):
    """Train a neural collision model on the given static world and return its checker.

    This will:
    - build a NeuralRobotCollisionSpherized from the exact model
    - train it on random configs for the provided world (spheres_batch)
    - expose a vmap'ed collision function with the same signature as make_collision_checker's output
    """

    # Wrap the world geometry in the same structure RobotCollisionSpherized expects
    # RobotCollisionSpherized.from_urdf constructs a CollGeom internally when used in examples,
    # so `spheres_batch` is already a valid batch of Sphere geometry.

    # Create neural collision model from existing exact model
    neural_coll = NeuralRobotCollisionSpherized.from_existing(robot_coll)

    # Train neural model on this specific world
    neural_coll = neural_coll.train(
        robot=robot,
        world_geom=spheres_batch,
        num_samples=10000,
        batch_size=256,  # Smaller batch = more gradient updates per epoch
        epochs=100,      # More epochs for better convergence
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

    print(f"Time to compute: {end_time - start_time:.6f} seconds")
    print(f"Collision distances shape: {dists.shape}")
    print(f"Min distance: {jnp.min(dists):.6f}")
    print(f"Max distance: {jnp.max(dists):.6f}")
    print(f"Mean distance: {jnp.mean(dists):.6f}")
    print(f"Std distance: {jnp.std(dists):.6f}")
    
    in_collision = dists < 0
    print(f"Number of collision pairs: {jnp.sum(in_collision)}")
    
    return dists

def main():
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
    q_batch = generate_configs(joints, 500000)

    # Create collision checker using exact model
    exact_check_collisions = make_collision_checker(robot, robot_coll)

    # Create and train neural collision checker for this specific world
    print("Training neural collision model (this may take a while)...")
    neural_check_collisions = make_neural_collision_checker(robot, robot_coll, spheres_batch)

    # Run benchmarks
    exact_dists = run_benchmark("Exact (RobotCollisionSpherized)", exact_check_collisions, q_batch, spheres_batch)
    neural_dists = run_benchmark("Neural (NeuralRobotCollisionSpherized)", neural_check_collisions, q_batch, spheres_batch)
    
    # Compare results
    print("\n=== Comparison ===")
    diff = neural_dists - exact_dists
    print(f"Mean absolute error: {jnp.mean(jnp.abs(diff)):.6f}")
    print(f"Max absolute error: {jnp.max(jnp.abs(diff)):.6f}")
    print(f"Mean error (bias): {jnp.mean(diff):.6f}")
    
    # Check accuracy at collision boundary
    exact_in_collision = exact_dists < 0
    neural_in_collision = neural_dists < 0
    
    true_positives = jnp.sum(exact_in_collision & neural_in_collision)
    false_positives = jnp.sum(~exact_in_collision & neural_in_collision)
    false_negatives = jnp.sum(exact_in_collision & ~neural_in_collision)
    true_negatives = jnp.sum(~exact_in_collision & ~neural_in_collision)
    
    print(f"\nCollision Detection Accuracy:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  True Negatives: {true_negatives}")
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
   

if __name__ == "__main__":
    main()
