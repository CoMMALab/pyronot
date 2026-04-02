"""IK with Collision — CUDA Least-Squares Solver with bench_env.json obstacles

Cover visualization: Panda robot doing IK with collision avoidance against
all obstacles loaded from bench_env.json (spheres rendered with randomized
vivid colors), with a three-point lighting rig.

The collision constraint penalizes penetration with the floor half-space and
every sphere in the environment simultaneously.  All obstacles are static
(no draggable handles), so the sphere geometry is closed over in the penalty
function and requires no constraint_args retrace.
"""

import json
import pathlib
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyronot as pk
import trimesh
import viser
from pyronot.collision import HalfSpace, RobotCollision, Sphere, collide
from pyronot.optimization_engines._ls_ik import ls_ik_solve_cuda
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def _random_vivid_rgba(rng: np.random.Generator, alpha: int = 210) -> np.ndarray:
    """Return a vivid random RGBA color using HSV sampling."""
    h = rng.uniform(0.0, 1.0)
    s = rng.uniform(0.7, 1.0)
    v = rng.uniform(0.75, 1.0)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb_map = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = rgb_map[i % 6]
    return np.array([int(r * 255), int(g * 255), int(b * 255), alpha], dtype=np.uint8)


def main():
    # ── Load bench environment ────────────────────────────────────────────────
    env_path = pathlib.Path("resources/bench_env_large.json")
    with env_path.open("r", encoding="utf-8") as f:
        env = json.load(f)
    env.setdefault("spheres", [])
    env.setdefault("cuboids", [])

    # ── Robot setup ───────────────────────────────────────────────────────────
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    hand_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    fixed_joint_mask = jnp.array(
        [name in hand_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.int32,
    )
    target_link_index = robot.links.names.index(target_link_name)

    # ── Build static collision objects from bench_env ─────────────────────────
    floor_data = env.get("floor", {})
    plane_coll = HalfSpace.from_point_and_normal(
        np.array(floor_data.get("point", [0.0, 0.0, 0.0])),
        np.array(floor_data.get("normal", [0.0, 0.0, 1.0])),
    )

    # One Sphere collision object per env sphere (static, closed over below).
    env_sphere_colls: list[Sphere] = [
        Sphere.from_center_and_radius(
            center=np.array(s["center"], dtype=np.float32),
            radius=np.array([float(s["radius"])]),
        )
        for s in env["spheres"]
    ]

    # ── Viser setup ───────────────────────────────────────────────────────────
    server = viser.ViserServer()

    # Three-point lighting rig (always on — this is the cover view).
    light_key = server.scene.add_light_point(
        "/lights/key",
        color=(255, 245, 220),
        intensity=40.0,
        position=(1.5, 1.0, 2.5),
        cast_shadow=True,
    )
    light_fill = server.scene.add_light_point(
        "/lights/fill",
        color=(200, 220, 255),
        intensity=20.0,
        position=(-1.5, 0.5, 1.5),
    )
    light_back = server.scene.add_light_point(
        "/lights/back",
        color=(255, 210, 180),
        intensity=15.0,
        position=(0.0, -1.5, 2.0),
    )

    with server.gui.add_folder("Lighting"):
        key_intensity = server.gui.add_slider(
            "Key intensity", min=0.0, max=80.0, step=0.5, initial_value=40.0
        )
        key_x = server.gui.add_slider(
            "Key x", min=-3.0, max=3.0, step=0.05, initial_value=1.5
        )
        key_y = server.gui.add_slider(
            "Key y", min=-3.0, max=3.0, step=0.05, initial_value=1.0
        )
        key_z = server.gui.add_slider(
            "Key z", min=0.1, max=4.0, step=0.05, initial_value=2.5
        )

        fill_intensity = server.gui.add_slider(
            "Fill intensity", min=0.0, max=80.0, step=0.5, initial_value=20.0
        )
        fill_x = server.gui.add_slider(
            "Fill x", min=-3.0, max=3.0, step=0.05, initial_value=-1.5
        )
        fill_y = server.gui.add_slider(
            "Fill y", min=-3.0, max=3.0, step=0.05, initial_value=0.5
        )
        fill_z = server.gui.add_slider(
            "Fill z", min=0.1, max=4.0, step=0.05, initial_value=1.5
        )

        back_intensity = server.gui.add_slider(
            "Back intensity", min=0.0, max=80.0, step=0.5, initial_value=15.0
        )
        back_x = server.gui.add_slider(
            "Back x", min=-3.0, max=3.0, step=0.05, initial_value=0.0
        )
        back_y = server.gui.add_slider(
            "Back y", min=-3.0, max=3.0, step=0.05, initial_value=-1.5
        )
        back_z = server.gui.add_slider(
            "Back z", min=0.1, max=4.0, step=0.05, initial_value=2.0
        )

    def _update_light_positions() -> None:
        light_key.position = (key_x.value, key_y.value, key_z.value)
        light_fill.position = (fill_x.value, fill_y.value, fill_z.value)
        light_back.position = (back_x.value, back_y.value, back_z.value)

    @key_intensity.on_update
    def _(_event) -> None:
        light_key.intensity = float(key_intensity.value)

    @fill_intensity.on_update
    def _(_event) -> None:
        light_fill.intensity = float(fill_intensity.value)

    @back_intensity.on_update
    def _(_event) -> None:
        light_back.intensity = float(back_intensity.value)

    @key_x.on_update
    @key_y.on_update
    @key_z.on_update
    @fill_x.on_update
    @fill_y.on_update
    @fill_z.on_update
    @back_x.on_update
    @back_y.on_update
    @back_z.on_update
    def _(_event) -> None:
        _update_light_positions()

    floor_z = float(floor_data.get("point", [0.0, 0.0, 0.0])[2])
    floor_thickness = 0.01
    floor_mesh = trimesh.creation.box(
        extents=(2.0, 2.0, floor_thickness),
        transform=trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, floor_z - floor_thickness / 2], dtype=np.float32)
        ),
    )
    floor_mesh.visual.face_colors = np.array([150, 150, 150, 255], dtype=np.uint8)
    server.scene.add_mesh_trimesh("/floor", floor_mesh)

    # Render env spheres with randomized vivid colors.
    rng = np.random.default_rng(seed=42)
    for i, s in enumerate(env["spheres"]):
        name = s.get("name", f"sphere_{i}")
        sphere_vis = Sphere.from_center_and_radius(
            center=np.array(s["center"], dtype=np.float32),
            radius=float(s["radius"]),
        )
        mesh = sphere_vis.to_trimesh()
        mesh.visual.face_colors = _random_vivid_rgba(rng)
        server.scene.add_mesh_trimesh(f"/env/spheres/{name}", mesh)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    show_target_transform = server.gui.add_checkbox(
        "Show target transform", initial_value=True
    )

    @show_target_transform.on_update
    def _(_event) -> None:
        ik_target_handle.visible = bool(show_target_transform.value)

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    pos_error_handle = server.gui.add_number(
        "Position error (mm)", 0.0, step=1e-9, disabled=True
    )

    # ── Collision constraint ──────────────────────────────────────────────────
    # env_sphere_colls is closed over — no retrace when the loop runs.
    _COLL_EPS = 0.005
    _coll_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)

    def _collision_penalty(cfg, robot, _unused):
        """Differentiable penalty: floor + all bench_env spheres."""
        coll_geom = robot_coll.at_config(robot, cfg)
        penalty = jnp.sum(
            jax.nn.softplus(
                -_coll_vs_world(coll_geom, plane_coll.broadcast_to((1,))) / _COLL_EPS
            ) * _COLL_EPS
        )
        for sphere_obs in env_sphere_colls:
            penalty += jnp.sum(
                jax.nn.softplus(
                    -_coll_vs_world(coll_geom, sphere_obs.broadcast_to((1,))) / _COLL_EPS
                ) * _COLL_EPS
            )
        return penalty

    constraints = [_collision_penalty]
    constraint_args = [None]
    constraint_weights = [1e8]

    rng_key = jax.random.PRNGKey(0)
    solution = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    while True:
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(ik_target_handle.wxyz)),
            translation=jnp.array(ik_target_handle.position),
        )

        rng_key, subkey = jax.random.split(rng_key)
        start_time = time.perf_counter()

        solution = ls_ik_solve_cuda(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=subkey,
            previous_cfg=solution,
            num_seeds=256,
            fixed_joint_mask=fixed_joint_mask,
            constraints=constraints,
            constraint_args=constraint_args,
            constraint_weights=constraint_weights,
        )
        solution.block_until_ready()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timing_handle.value = elapsed_ms

        link_poses = robot.forward_kinematics(solution)
        actual_pose = jaxlie.SE3(link_poses[target_link_index])
        pos_error = jnp.linalg.norm(
            actual_pose.translation() - target_pose.translation()
        )
        pos_error_handle.value = float(pos_error) * 1000  # mm

        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
