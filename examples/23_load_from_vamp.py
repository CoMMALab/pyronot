import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as onp
import jax
import jax.numpy as jnp
import yourdfpy
import pyronot_snippets as pks

from pyronot.collision._geometry import Sphere, Capsule, HalfSpace, Box
from pyronot.collision._geometry import Box 
from pyronot.collision._robot_collision import RobotCollision, RobotCollisionSpherized
from pyronot._robot import Robot
from pyronot.collision._utils import *
from pyronot.collision._obstacles import *


import argparse
import pickle
from pathlib import Path
import time
import viser
from viser.extras import ViserUrdf
import trimesh
import hydra
from omegaconf import DictConfig
import numpy as np

def get_configs_path():
    return Path(__file__).parent.parent / "configs"

def get_data_path():
    return Path(__file__).parent.parent / "data"

def load_robot_urdf(urdf_file: Path):
    urdf_path = get_data_path() / urdf_file
    assert urdf_path.exists(), f"URDF file not found at {urdf_path}"
    return yourdfpy.URDF.load(urdf_path.as_posix(), mesh_dir=urdf_path.parent.as_posix())

def visualize_obstacles_with_viser(obstacles, urdf=None, initial_config=None, block: bool = True):
    """
    Visualize a list of obstacle geometry objects (HalfSpace, Capsule, Sphere)
    using the Viser 3D viewer.

    Args:
        obstacles: List of geometry objects with `.pose` and `.size` attributes.
        urdf: Optional yourdfpy.URDF object to visualize.
        initial_config: Optional initial configuration for the robot.
    """
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", show_axes=True)
    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_light_hemisphere("/lights/ambient", intensity=0.6)
    server.scene.add_light_directional("/lights/key", intensity=2.0, cast_shadow=True)

    if urdf is not None:
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
        if initial_config is not None:
            urdf_vis.update_cfg(np.array(initial_config))
        ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    for i, obs in enumerate(obstacles):
        if hasattr(obs, "to_trimesh"):
            mesh = obs.to_trimesh()
            server.scene.add_mesh_trimesh(name=f"/world/obstacles/obj_{i}", mesh=mesh, visible=True)
            continue

        # Unknown obstacle: place a small yellow icosphere at origin
        server.scene.add_icosphere(name=f"/world/obstacles/unknown_{i}", radius=0.05, position=(0, 0, 0), color=(255, 255, 0))

        
    print("Viser server started. Open the URL printed above to view obstacles.")


    if not block:
        return server

    print("Viser running — press Ctrl+C to stop.")



    try:
        while True:
            solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target_handle.position),
            target_wxyz=np.array(ik_target_handle.wxyz),
            )
            # print("Cost at solution:", cost)
            urdf_vis.update_cfg(solution)
    except KeyboardInterrupt:
        print("Stopping viser server...")



@hydra.main(version_base=None, config_path=get_configs_path().as_posix(), config_name="demo_gtmp_mbm")
def main(cfg: DictConfig):
    global robot, robot_coll, target_link_name
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    robot = cfg.robot
    problem = cfg.problem
    index = cfg.index

    target_link_name = "panda_hand"

    data_dir = get_data_path() / f"{robot}"
    with open(data_dir / "problems.pkl", 'rb') as f:
        data = pickle.load(f)

    if not problem:
        problem = list(data['problems'].keys())[0]

    if problem not in data['problems']:
        raise RuntimeError(
            f"""No problem with name {problem}!
                Existing problems: {list(data['problems'].keys())}"""
        )

    problems = data['problems'][problem]
    try:
        problem_data = next(problem for problem in problems if problem['index'] == index)
    except StopIteration:
        raise RuntimeError(f"No problem in {problem} with index {index}!")

    start = jnp.array(problem_data['start'])
    goals = jnp.array(problem_data['goals'])
    valid = problem_data['valid']

    print(start, goals, valid)

    # Load Panda Robot
    urdf = load_robot_urdf("panda/panda_spherized.urdf")
    robot = Robot.from_urdf(urdf)
    print(f"Robot methods: {dir(robot)}")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)

    obstacles = create_collision_environment(problem_data)
    print(f"Created {len(obstacles)} collision objects.")
    for i in range(len(obstacles)):
        print(obstacles[i])
    visualize_obstacles_with_viser(obstacles, urdf=urdf, initial_config=start)

if __name__ == "__main__":
    main()
