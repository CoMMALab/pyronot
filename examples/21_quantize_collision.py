"""Quantize Collision Example

Script to generate obstacles and robot configurations for quantization testing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import pyronot as pk
from pyronot.collision import RobotCollisionSpherized, Sphere
from pyronot._robot_urdf_parser import RobotURDFParser
import yourdfpy
from pyronot.utils import quantize


def generate_spheres(n_spheres):
    print(f"Generating {n_spheres} random spheres...")
    spheres = []
    for _ in range(n_spheres):
        center = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        radius = np.random.uniform(low=0.05, high=0.2)
        sphere = Sphere.from_center_and_radius(center, np.array([radius]))
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
        size=(n_configs, joints.num_actuated_joints),
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


def run_benchmark(name, check_fn, q_batch, obstacles):
    print(f"\n{name}:")

    # Metrics
    q_size_mb = q_batch.nbytes / 1024 / 1024
    spheres_size_mb = (
        sum(x.nbytes for x in jax.tree_util.tree_leaves(obstacles)) / 1024 / 1024
    )

    print(f"q_batch size: {q_size_mb:.2f} MB")
    print(f"Obstacles (spheres) size: {spheres_size_mb:.2f} MB")

    # Warmup
    print(f"Warming up JIT ({name})...")
    _ = check_fn(q_batch, obstacles)
    # _.block_until_ready()

    # Run collision checking
    print(f"Executing collision checking ({name})...")
    start_time = time.perf_counter()
    dists = check_fn(q_batch, obstacles)
    # dists.block_until_ready()
    end_time = time.perf_counter()

    print(f"Time to compute: {end_time - start_time:.6f} seconds")
    print(f"Collision distances shape: {dists.shape}")
    print(f"Min distance: {jnp.min(dists)}")

    in_collision = dists < 0
    print(f"Number of collision pairs: {jnp.sum(in_collision)}")


def main():
    global robot, robot_coll
    # Load robot
    urdf_path = "resources/ur5/ur5_spherized.urdf"
    mesh_dir = "resources/ur5/meshes"
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    joints, links = RobotURDFParser.parse(urdf)

    # Initialize collision model
    print("Initializing collision model...")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    # Create collision checker
    check_collisions = make_collision_checker(robot, robot_coll)

    # Generate data
    spheres_batch = generate_spheres(10000)
    q_batch = generate_configs(joints, 10000)

    # Run benchmarks
    run_benchmark("Default (float32)", check_collisions, q_batch, spheres_batch)

    # Quantized
    q_batch_f16 = quantize(q_batch)
    spheres_batch_f16 = quantize(spheres_batch)
    run_benchmark(
        "Quantized (float16)", check_collisions, q_batch_f16, spheres_batch_f16
    )

    q_batch_int8 = quantize(q_batch, jax.numpy.int8)
    spheres_batch_int8 = quantize(spheres_batch, jax.numpy.int8)
    run_benchmark(
        "Quantized (int8)", check_collisions, q_batch_int8, spheres_batch_int8
    )


if __name__ == "__main__":
    main()
