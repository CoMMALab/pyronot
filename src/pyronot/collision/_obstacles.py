import jax
import jax.numpy as jnp
import numpy as np
from pyronot.collision._geometry import Sphere, Capsule, HalfSpace, Box
from collections import defaultdict


def euler_to_quaternion(euler_xyz):
    """Convert Euler angles (roll, pitch, yaw) in radians to quaternion (w, x, y, z).

    Args:
        euler_xyz: array-like of length 3 (roll, pitch, yaw)

    Returns:
        jnp.ndarray shape (4,) in (w, x, y, z) order.
    """
    roll, pitch, yaw = jnp.asarray(euler_xyz, dtype=jnp.float32)

    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.array([w, x, y, z], dtype=jnp.float32)


import jax.numpy as jnp

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        quat: array-like length 4 (w, x, y, z)

    Returns:
        jnp.ndarray shape (3, 3)
    """
    q = jnp.asarray(quat, dtype=jnp.float32)
    w, x, y, z = q

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = jnp.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
    ], dtype=jnp.float32)

    return R




def create_collision_environment(problem_data):
    """
    Create collision geometry objects from problem data dictionary.
    
    Args:
        problem_data: Dictionary containing obstacle definitions:
            - "sphere": List of dicts with "position" and "radius"
            - "cylinder": List of dicts with "position", "radius", "length", "orientation_euler_xyz"
            - "box": List of dicts with "position", "half_extents", "orientation_euler_xyz"
            
    Returns:
        List of collision geometry objects
    """
    obstacles = []
    
    
    ground_normal = jnp.array([[0.0, 0.0, 1.0]])
    ground_point = jnp.array([[0.0, 0.0, -1.0]])
    ground_plane = HalfSpace.from_point_and_normal(
        point=ground_point,
        normal=ground_normal
    )
    obstacles.append(ground_plane)
    
    
    for sphere_data in problem_data.get("sphere", []):
        pos = jnp.array([sphere_data["position"]])  
        radius = jnp.array([sphere_data["radius"]])  
        
        sphere = Sphere.from_center_and_radius(
            center=pos,
            radius=radius
        )
        obstacles.append(sphere)
    
    
    for cyl_data in problem_data.get("cylinder", []):
        pos = jnp.array(cyl_data["position"])  
        radius = jnp.array([cyl_data["radius"]])
        length = jnp.array([cyl_data["length"]])
        euler = cyl_data["orientation_euler_xyz"]
        
        # Convert Euler to quaternion and rotation matrix
        quat = euler_to_quaternion(euler)
        rot_mat = quaternion_to_rotation_matrix(quat)
        
        # Capsule local axis (z-axis)
        local_axis = jnp.array([0.0, 0.0, 1.0])
        world_axis = rot_mat @ local_axis
        
        # Define capsule using radius/height, centered at position
        capsule = Capsule.from_radius_height(
            radius=radius,
            height=length,
            position=pos.reshape(1, 3)  # Must be batched (N, 3)
        )
        
        obstacles.append(capsule)
    
    for box_data in problem_data.get("box", []):
        pos = jnp.array(box_data["position"])       # box center
        half_ext = jnp.array(box_data["half_extents"])  # half-lengths along x, y, z

        length = float(2.0 * half_ext[0])
        width = float(2.0 * half_ext[1])
        height = float(2.0 * half_ext[2])


        # Optional orientation: support a few possible keys safely
        wxyz = None
        if "orientation_quat_xyzw" in box_data and box_data["orientation_quat_xyzw"] is not None:
            qarr = np.asarray(box_data["orientation_quat_xyzw"]).reshape(-1)[:4]
            wxyz = (float(qarr[3]), float(qarr[0]), float(qarr[1]), float(qarr[2]))

        if wxyz is not None:
            box = Box.from_center_and_dimensions(center=pos, length=length, width=width, height=height, wxyz=wxyz)
        else:
            box = Box.from_center_and_dimensions(center=pos, length=length, width=width, height=height)
       


        obstacles.append(box)

    
    return obstacles

def stack_obstacles(obstacles):
    obs_by_type = defaultdict(list)
    for obs in obstacles:
        obs_by_type[type(obs)].append(obs)
    
    stacked_obstacles = []
    for obs_type, obs_list in obs_by_type.items():
        # Stack leaves of the pytrees
        # For primitive types (Sphere, Box, Capsule, HalfSpace), the structure is consistent.
        try:
            stacked_obs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *obs_list)
            stacked_obstacles.append(stacked_obs)
        except Exception as e:
            print(f"Failed to stack obstacles of type {obs_type}: {e}")
            raise e
    
    return tuple(stacked_obstacles)
