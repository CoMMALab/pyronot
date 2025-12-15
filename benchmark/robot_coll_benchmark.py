import numpy as np 
import pyronot as prn
from pyronot.collision import Sphere, RobotCollision, RobotCollisionSpherized
import yourdfpy
import pinocchio as pin
import hppfcl
import time 

np.random.seed(42)

NUM_SAMPLES = 1000

SPHERE_CENTERS = [
    [0.55, 0, 0.25],
    [0.35, 0.35, 0.25],
    [0, 0.55, 0.25],
    [-0.55, 0, 0.25],
    [-0.35, -0.35, 0.25],
    [0, -0.55, 0.25],
    [0.35, -0.35, 0.25],
    [0.35, 0.35, 0.8],
    [0, 0.55, 0.8],
    [-0.35, 0.35, 0.8],
    [-0.55, 0, 0.8],
    [-0.35, -0.35, 0.8],
    [0, -0.55, 0.8],
    [0.35, -0.35, 0.8],
    ]

SPHERE_R = [0.2] * len(SPHERE_CENTERS)

sphere_coll = Sphere.from_center_and_radius(SPHERE_CENTERS, SPHERE_R)

urdf_path = "resources/ur5/ur5_spherized.urdf"
mesh_dir = "resources/ur5/meshes"
urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
robot = prn.Robot.from_urdf(urdf)
robot_coll_capsule = RobotCollision.from_urdf(urdf)
robot_coll_sphere = RobotCollisionSpherized.from_urdf(urdf)


# ============ Pinocchio Ground Truth Setup ============
def setup_pinocchio_collision(urdf_path, mesh_dir, sphere_centers, sphere_radii):
    """Setup Pinocchio collision model with environment spheres."""
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_geom_model = pin.buildGeomFromUrdf(pin_model, urdf_path, pin.COLLISION, mesh_dir)
    
    num_robot_geoms = len(pin_geom_model.geometryObjects)
    
    # Add obstacle spheres to the geometry model
    for i, (center, radius) in enumerate(zip(sphere_centers, sphere_radii)):
        sphere_shape = hppfcl.Sphere(radius)
        placement = pin.SE3(np.eye(3), np.array(center))
        geom_obj = pin.GeometryObject(
            f"obstacle_sphere_{i}",
            0,  # parent frame (universe)
            0,  # parent joint (universe)
            sphere_shape,
            placement,
        )
        pin_geom_model.addGeometryObject(geom_obj)
    
    # Add collision pairs: robot geometries vs obstacle spheres
    num_obstacles = len(sphere_centers)
    for robot_geom_id in range(num_robot_geoms):
        for obs_idx in range(num_obstacles):
            obs_geom_id = num_robot_geoms + obs_idx
            pin_geom_model.addCollisionPair(
                pin.CollisionPair(robot_geom_id, obs_geom_id)
            )
    
    pin_data = pin_model.createData()
    pin_geom_data = pin.GeometryData(pin_geom_model)
    
    return pin_model, pin_data, pin_geom_model, pin_geom_data


def check_collision_pinocchio(pin_model, pin_data, pin_geom_model, pin_geom_data, q):
    """Check if configuration q is in collision using Pinocchio (ground truth)."""
    pin.updateGeometryPlacements(pin_model, pin_data, pin_geom_model, pin_geom_data, q)
    return pin.computeCollisions(pin_geom_model, pin_geom_data, True)


# Setup Pinocchio collision checker
pin_model, pin_data, pin_geom_model, pin_geom_data = setup_pinocchio_collision(
    urdf_path, mesh_dir, SPHERE_CENTERS, SPHERE_R
)


def generate_dataset(num_samples):
    q_batch = []
    robot_lower_limits = robot.joints.lower_limits
    robot_upper_limits = robot.joints.upper_limits
    for _ in range(num_samples):
        q = np.random.uniform(robot_lower_limits, robot_upper_limits)
        q_batch.append(q)
    return np.array(q_batch)

q_batch = generate_dataset(NUM_SAMPLES)
print(f"Generated {q_batch.shape[0]} samples")
# Warmup for JIT 
q = q_batch[0]
robot_coll_capsule.at_config(robot, q)
robot_coll_capsule.compute_world_collision_distance(robot, q, sphere_coll)
robot_coll_sphere.at_config(robot, q)
robot_coll_sphere.compute_world_collision_distance(robot, q, sphere_coll)

# ============ Generate Ground Truth with Pinocchio ============
print("\n=== Generating Pinocchio Ground Truth ===")
start_time = time.time()
ground_truth = []
for q in q_batch:
    collision = check_collision_pinocchio(pin_model, pin_data, pin_geom_model, pin_geom_data, q)
    ground_truth.append(collision)
ground_truth = np.array(ground_truth)
end_time = time.time()
time_taken_ms = (end_time - start_time) * 1000
print(f"Time taken for Pinocchio ground truth (ms): {time_taken_ms:.2f}")
print(f"Time per collision check (ms): {time_taken_ms/NUM_SAMPLES:.4f}")
print(f"Collision rate: {ground_truth.sum()}/{NUM_SAMPLES} ({100*ground_truth.mean():.1f}%)")

# ============ Benchmark pyronot collision methods ============
print("\n=== Benchmarking pyronot Collision Methods ===")

start_time = time.time()
for q in q_batch:
    robot_coll_sphere.at_config(robot, q)
    robot_coll_sphere.compute_world_collision_distance(robot, q, sphere_coll)
end_time = time.time()
time_taken_ms = (end_time - start_time) * 1000
print(f"Time taken for sphere for single collision check (ms): {time_taken_ms/NUM_SAMPLES:.4f}")

start_time = time.time()
for q in q_batch:
    robot_coll_capsule.at_config(robot, q)
    robot_coll_capsule.compute_world_collision_distance(robot, q, sphere_coll)
end_time = time.time()
time_taken_ms = (end_time - start_time) * 1000
print(f"Time taken for capsule for single collision check (ms): {time_taken_ms/NUM_SAMPLES:.4f}")

# ============ Compare with Ground Truth ============
print("\n=== Accuracy Comparison with Ground Truth ===")

# Check sphere model accuracy
sphere_predictions = []
for q in q_batch:
    dist = robot_coll_sphere.compute_world_collision_distance(robot, q, sphere_coll)
    # Collision if any distance is negative
    in_collision = (np.array(dist) < 0).any()
    sphere_predictions.append(in_collision)
sphere_predictions = np.array(sphere_predictions)
sphere_accuracy = (sphere_predictions == ground_truth).mean()
print(f"Sphere model accuracy: {100*sphere_accuracy:.2f}%")

# Check capsule model accuracy  
capsule_predictions = []
for q in q_batch:
    dist = robot_coll_capsule.compute_world_collision_distance(robot, q, sphere_coll)
    # Collision if any distance is negative
    in_collision = (np.array(dist) < 0).any()
    capsule_predictions.append(in_collision)
capsule_predictions = np.array(capsule_predictions)
capsule_accuracy = (capsule_predictions == ground_truth).mean()
print(f"Capsule model accuracy: {100*capsule_accuracy:.2f}%")

