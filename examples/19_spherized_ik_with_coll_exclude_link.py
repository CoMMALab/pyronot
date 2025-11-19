"""IK with Collision

Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import time

import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, RobotCollisionSpherized, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import pyroki_snippets as pks
import yourdfpy 


def main():
    """Main function for basic IK with collision."""
    urdf_path = "resources/ur5/ur5_spherized.urdf"
    mesh_dir = "resources/ur5/meshes"
    target_link_name = "robotiq_85_tool_link"

    # urdf_path = "resources/panda/panda_spherized.urdf"
    # mesh_dir = "resources/panda/meshes"
    # target_link_name = "panda_hand"
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=[0, -1.57, 1.57, -1.57, -1.57, 0])

    robot_coll = RobotCollisionSpherized.from_urdf(urdf)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target.
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.0, 0.6, 0.2), wxyz=(0, 0, 1, 0)
    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    exclude_links_from_cc = ["offset_link", "base_link"]
    exclude_link_mask = robot.links.get_link_mask_from_names(exclude_links_from_cc)
    print(exclude_link_mask)
    while True:
        start_time = time.time()

        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        world_coll_list = [plane_coll, sphere_coll_world_current]
        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
        )

        # Update timing handle.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizer.
        urdf_vis.update_cfg(solution)
        # print(robot.links.names)
        # Compute the collision of the solution 
        distance_link_to_plane = robot_coll.compute_world_collision_distance(
            robot, 
            solution,
            plane_coll
        )
        distance_link_to_plane = RobotCollisionSpherized.mask_collision_distance(distance_link_to_plane, exclude_link_mask)
        # print(distance_link_to_plane)
        distance_link_to_sphere = robot_coll.compute_world_collision_distance(
            robot, 
            solution,
            sphere_coll
        )
        distance_link_to_sphere = RobotCollisionSpherized.mask_collision_distance(distance_link_to_sphere, exclude_link_mask)
        # print(distance_link_to_sphere)
        # Visualize collision representation
        robot_coll_config: Sphere = robot_coll.at_config(robot, solution)
        # print(robot_coll_config.get_batch_axes()[-1])
        robot_coll_mesh = robot_coll_config.to_trimesh()
        server.scene.add_mesh_trimesh(
            "/robot_coll", 
            mesh=robot_coll_mesh,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    main()
