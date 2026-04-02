"""Visualize a benchmark environment JSON in viser.

Usage:
    python scripts/visualize_bench_env.py
    python scripts/visualize_bench_env.py --env resources/bench_env.json --port 8080
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyronot as pk
import viser
import yourdfpy

from pyronot.collision import Box, RobotCollisionSpherized, Sphere


def _random_vivid_rgba(rng: np.random.Generator, alpha: int = 210) -> np.ndarray:
    """Return a vivid random RGBA color using HSV sampling."""
    h = rng.uniform(0.0, 1.0)
    s = rng.uniform(0.7, 1.0)
    v = rng.uniform(0.75, 1.0)
    # HSV -> RGB (manual, no extra import)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb_map = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = rgb_map[i % 6]
    return np.array([int(r * 255), int(g * 255), int(b * 255), alpha], dtype=np.uint8)


def _load_env(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scene_extent_xy(env: dict) -> float:
    pts: list[np.ndarray] = []

    for s in env.get("spheres", []):
        c = np.array(s["center"], dtype=np.float32)
        r = float(s["radius"])
        pts.extend([
            c + np.array([r, 0.0, 0.0], dtype=np.float32),
            c + np.array([-r, 0.0, 0.0], dtype=np.float32),
            c + np.array([0.0, r, 0.0], dtype=np.float32),
            c + np.array([0.0, -r, 0.0], dtype=np.float32),
        ])

    for b in env.get("cuboids", []):
        c = np.array(b["center"], dtype=np.float32)
        d = np.array(b["dims"], dtype=np.float32) * 0.5
        pts.extend([
            c + np.array([d[0], d[1], 0.0], dtype=np.float32),
            c + np.array([d[0], -d[1], 0.0], dtype=np.float32),
            c + np.array([-d[0], d[1], 0.0], dtype=np.float32),
            c + np.array([-d[0], -d[1], 0.0], dtype=np.float32),
        ])

    if not pts:
        return 2.0

    arr = np.stack(pts)
    max_abs_xy = float(np.max(np.abs(arr[:, :2])))
    return max(2.0, 2.0 * max_abs_xy + 0.5)


def _discover_spherized_urdfs(resources_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    urdf_paths = sorted(resources_dir.rglob("*.urdf"))
    out: dict[str, pathlib.Path] = {}
    collisions: dict[str, list[pathlib.Path]] = {}

    for urdf_path in urdf_paths:
        stem = urdf_path.stem
        if not stem.endswith("_spherized"):
            continue
        robot_name = stem[: -len("_spherized")]
        if robot_name in out:
            collisions.setdefault(robot_name, [out[robot_name]]).append(urdf_path)
            continue
        out[robot_name] = urdf_path

    if collisions:
        msg_lines = []
        for robot_name, paths in collisions.items():
            msg_lines.append(f"{robot_name}: {[str(p) for p in paths]}")
        raise ValueError(
            "Found duplicate spherized URDFs for robot names:\n" + "\n".join(msg_lines)
        )
    return out


def _default_cfg_from_limits(robot: pk.Robot) -> np.ndarray:
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
    finite = np.isfinite(lower) & np.isfinite(upper)
    return np.where(finite, 0.5 * (lower + upper), 0.0).astype(np.float32)


def _render_robot_primitives(
    server: viser.ViserServer,
    robot: pk.Robot,
    robot_coll: RobotCollisionSpherized,
    cfg: np.ndarray,
) -> int:
    coll_world = robot_coll.at_config(robot, cfg)
    if not isinstance(coll_world, Sphere):
        raise TypeError("Expected RobotCollisionSpherized.at_config to return Sphere geometry.")

    batch_axes = coll_world.get_batch_axes()
    if len(batch_axes) < 2:
        raise ValueError(f"Unexpected robot collision batch axes: {batch_axes}")

    num_spheres_per_link, num_links = batch_axes[-2], batch_axes[-1]
    centers = np.asarray(coll_world.pose.translation(), dtype=np.float32)
    radii = np.asarray(coll_world.radius, dtype=np.float32)
    blue_rgba = np.array([50, 120, 255, 170], dtype=np.uint8)

    n_rendered = 0
    for link_idx in range(num_links):
        link_name = robot_coll.link_names[link_idx].replace("/", "_")
        for sphere_idx in range(num_spheres_per_link):
            radius = float(radii[sphere_idx, link_idx])
            if radius <= 0.0:
                continue  # Skip padded/degenerate spheres.
            center = centers[sphere_idx, link_idx]
            sphere = Sphere.from_center_and_radius(center=center, radius=radius)
            mesh = sphere.to_trimesh()
            mesh.visual.face_colors = blue_rgba
            server.scene.add_mesh_trimesh(
                f"/robot_primitives/{link_name}/sphere_{sphere_idx}",
                mesh=mesh,
            )
            n_rendered += 1
    return n_rendered


def main() -> None:
    resources_dir = pathlib.Path("resources")
    robot_models = _discover_spherized_urdfs(resources_dir)
    if not robot_models:
        raise FileNotFoundError(
            f"No *_spherized.urdf files found under {resources_dir.resolve()}."
        )

    default_robot = "ur5" if "ur5" in robot_models else sorted(robot_models.keys())[0]

    parser = argparse.ArgumentParser(description="Visualize bench_env.json in viser.")
    parser.add_argument(
        "--env",
        type=pathlib.Path,
        default=pathlib.Path("resources/bench_env.json"),
        help="Path to environment JSON.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--robot",
        type=str,
        default=default_robot,
        choices=tuple(sorted(robot_models.keys())),
        help="Robot name. Loads resources/**/<robot>_spherized.urdf.",
    )
    parser.add_argument(
        "--no-robot-primitives",
        action="store_true",
        default=False,
        help="Disable rendering of spherized URDF primitive geometry.",
    )
    args = parser.parse_args()

    env = _load_env(args.env)
    env.setdefault("spheres", [])
    env.setdefault("cuboids", [])

    robot_urdf_path = robot_models[args.robot]
    robot_mesh_dir = robot_urdf_path.parent / "meshes"
    if robot_mesh_dir.exists():
        urdf = yourdfpy.URDF.load(robot_urdf_path.as_posix(), mesh_dir=robot_mesh_dir.as_posix())
    else:
        urdf = yourdfpy.URDF.load(robot_urdf_path.as_posix())
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf)
    default_cfg = _default_cfg_from_limits(robot)

    rng = np.random.default_rng(seed=42)
    sphere_colors: dict[str, np.ndarray] = {}
    for i, s in enumerate(env.get("spheres", [])):
        name = s.get("name", f"sphere_{i}")
        sphere_colors[name] = _random_vivid_rgba(rng)

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")
    print(f"Loaded env: {args.env}")
    print(f"Loaded robot: {args.robot} ({robot_urdf_path})")

    # ------------------------------------------------------------------
    # Fancy lighting (initially hidden; toggled via GUI checkbox)
    # ------------------------------------------------------------------
    light_key = server.scene.add_light_point(
        "/lights/key",
        color=(255, 245, 220),
        intensity=40.0,
        position=(1.5, 1.0, 2.5),
        cast_shadow=True,
        visible=False,
    )
    light_fill = server.scene.add_light_point(
        "/lights/fill",
        color=(200, 220, 255),
        intensity=20.0,
        position=(-1.5, 0.5, 1.5),
        visible=False,
    )
    light_back = server.scene.add_light_point(
        "/lights/back",
        color=(255, 210, 180),
        intensity=15.0,
        position=(0.0, -1.5, 2.0),
        visible=False,
    )

    floor = env.get("floor", {})
    floor_pt = np.array(floor.get("point", [0.0, 0.0, 0.0]), dtype=np.float32)
    floor_n = np.array(floor.get("normal", [0.0, 0.0, 1.0]), dtype=np.float32)

    if np.linalg.norm(floor_n) > 1e-8:
        floor_n = floor_n / np.linalg.norm(floor_n)

    grid_size = _scene_extent_xy(env)
    if np.allclose(floor_n, np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
        server.scene.add_grid(
            "/env/floor_grid",
            width=grid_size,
            height=grid_size,
            cell_size=0.1,
            position=(0.0, 0.0, float(floor_pt[2])),
        )
    else:
        print("Warning: non-z-up floor normal; showing world-aligned grid only.")
        server.scene.add_grid("/env/floor_grid", width=grid_size, height=grid_size, cell_size=0.1)

    for i, s in enumerate(env.get("spheres", [])):
        name = s.get("name", f"sphere_{i}")
        center = np.array(s["center"], dtype=np.float32)
        radius = float(s["radius"])
        sphere = Sphere.from_center_and_radius(center=center, radius=radius)
        mesh = sphere.to_trimesh()
        mesh.visual.face_colors = sphere_colors[name]
        server.scene.add_mesh_trimesh(f"/env/spheres/{name}", mesh)

    for i, b in enumerate(env.get("cuboids", [])):
        name = b.get("name", f"cuboid_{i}")
        center = np.array(b["center"], dtype=np.float32)
        dims = np.array(b["dims"], dtype=np.float32)
        wxyz = np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        box = Box.from_center_and_dimensions(
            center=center,
            length=float(dims[0]),
            width=float(dims[1]),
            height=float(dims[2]),
            wxyz=wxyz,
        )
        server.scene.add_mesh_trimesh(f"/env/cuboids/{name}", box.to_trimesh())

    robot_primitives_frame = server.scene.add_frame(
        "/robot_primitives",
        show_axes=False,
        visible=not args.no_robot_primitives,
    )
    if not args.no_robot_primitives:
        num_robot_spheres = _render_robot_primitives(server, robot, robot_coll, default_cfg)
        print(f"Rendered {num_robot_spheres} robot collision spheres (blue).")
    else:
        print("Robot primitive geometry rendering disabled (--no-robot-primitives).")

    def _obstacle_options() -> list[str]:
        out: list[str] = []
        for i, s in enumerate(env["spheres"]):
            out.append(f"sphere::{s.get('name', f'sphere_{i}')}")
        for i, b in enumerate(env["cuboids"]):
            out.append(f"cuboid::{b.get('name', f'cuboid_{i}')}")
        return out

    def _parse_selection(selection: str) -> tuple[str, int]:
        kind, name = selection.split("::", 1)
        if kind == "sphere":
            for i, s in enumerate(env["spheres"]):
                if s.get("name", f"sphere_{i}") == name:
                    return kind, i
        elif kind == "cuboid":
            for i, b in enumerate(env["cuboids"]):
                if b.get("name", f"cuboid_{i}") == name:
                    return kind, i
        raise ValueError(f"Unknown selection: {selection}")

    def _render_obstacle(kind: str, idx: int) -> None:
        if kind == "sphere":
            s = env["spheres"][idx]
            name = s.get("name", f"sphere_{idx}")
            if name not in sphere_colors:
                sphere_colors[name] = _random_vivid_rgba(rng)
            center = np.array(s["center"], dtype=np.float32)
            radius = float(s["radius"])
            sphere = Sphere.from_center_and_radius(center=center, radius=radius)
            mesh = sphere.to_trimesh()
            mesh.visual.face_colors = sphere_colors[name]
            server.scene.add_mesh_trimesh(f"/env/spheres/{name}", mesh)
            return

        b = env["cuboids"][idx]
        name = b.get("name", f"cuboid_{idx}")
        center = np.array(b["center"], dtype=np.float32)
        dims = np.array(b["dims"], dtype=np.float32)
        wxyz = np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        box = Box.from_center_and_dimensions(
            center=center,
            length=float(dims[0]),
            width=float(dims[1]),
            height=float(dims[2]),
            wxyz=wxyz,
        )
        server.scene.add_mesh_trimesh(f"/env/cuboids/{name}", box.to_trimesh())

    def _render_all_obstacles() -> None:
        for i in range(len(env["spheres"])):
            _render_obstacle("sphere", i)
        for i in range(len(env["cuboids"])):
            _render_obstacle("cuboid", i)

    with server.gui.add_folder("Obstacle Editor"):
        options = _obstacle_options()
        if not options:
            env["spheres"].append({
                "name": "sphere_0",
                "center": [0.4, 0.0, 0.5],
                "radius": 0.1,
            })
            options = _obstacle_options()
            _render_all_obstacles()

        obstacle_select = server.gui.add_dropdown(
            "Selected obstacle", options=options, initial_value=options[0]
        )
        sphere_radius = server.gui.add_slider(
            "Sphere radius", min=0.01, max=0.5, step=0.005, initial_value=0.1
        )
        cuboid_length = server.gui.add_slider(
            "Cuboid length", min=0.02, max=1.0, step=0.01, initial_value=0.3
        )
        cuboid_width = server.gui.add_slider(
            "Cuboid width", min=0.02, max=1.0, step=0.01, initial_value=0.3
        )
        cuboid_height = server.gui.add_slider(
            "Cuboid height", min=0.02, max=1.0, step=0.01, initial_value=0.3
        )
        add_sphere_btn = server.gui.add_button("Add Sphere")
        add_cuboid_btn = server.gui.add_button("Add Cuboid")
        save_btn = server.gui.add_button("Save Env JSON")

    with server.gui.add_folder("Display"):
        fancy_lighting_cb = server.gui.add_checkbox("Fancy lighting", initial_value=False)
        show_primitives_cb = server.gui.add_checkbox(
            "Robot primitives", initial_value=not args.no_robot_primitives
        )

    @fancy_lighting_cb.on_update
    def _(_event) -> None:
        v = bool(fancy_lighting_cb.value)
        light_key.visible = v
        light_fill.visible = v
        light_back.visible = v

    @show_primitives_cb.on_update
    def _(_event) -> None:
        robot_primitives_frame.visible = bool(show_primitives_cb.value)

    _guard = {"syncing": False}
    transform_handle = server.scene.add_transform_controls(
        "/env/selected_transform",
        scale=0.18,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    def _sync_controls_from_selection() -> None:
        _guard["syncing"] = True
        kind, idx = _parse_selection(str(obstacle_select.value))
        if kind == "sphere":
            s = env["spheres"][idx]
            sphere_radius.visible = True
            cuboid_length.visible = False
            cuboid_width.visible = False
            cuboid_height.visible = False
            sphere_radius.value = float(s["radius"])
        else:
            b = env["cuboids"][idx]
            d = b["dims"]
            sphere_radius.visible = False
            cuboid_length.visible = True
            cuboid_width.visible = True
            cuboid_height.visible = True
            cuboid_length.value = float(d[0])
            cuboid_width.value = float(d[1])
            cuboid_height.value = float(d[2])
            transform_handle.wxyz = tuple(np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32))
        if kind == "sphere":
            center = np.array(env["spheres"][idx]["center"], dtype=np.float32)
            transform_handle.wxyz = (1.0, 0.0, 0.0, 0.0)
        else:
            center = np.array(env["cuboids"][idx]["center"], dtype=np.float32)
        transform_handle.position = tuple(center)
        _guard["syncing"] = False

    @transform_handle.on_update
    def _(_event) -> None:
        if _guard["syncing"]:
            return
        kind, idx = _parse_selection(str(obstacle_select.value))
        pos = np.array(transform_handle.position, dtype=np.float32).tolist()
        if kind == "sphere":
            env["spheres"][idx]["center"] = pos
            _render_obstacle("sphere", idx)
            return
        env["cuboids"][idx]["center"] = pos
        env["cuboids"][idx]["wxyz"] = np.array(transform_handle.wxyz, dtype=np.float32).tolist()
        _render_obstacle("cuboid", idx)

    @obstacle_select.on_update
    def _(_event) -> None:
        _sync_controls_from_selection()

    @sphere_radius.on_update
    def _(_event) -> None:
        if _guard["syncing"]:
            return
        kind, idx = _parse_selection(str(obstacle_select.value))
        if kind != "sphere":
            return
        env["spheres"][idx]["radius"] = float(sphere_radius.value)
        _render_obstacle("sphere", idx)

    @cuboid_length.on_update
    def _(_event) -> None:
        if _guard["syncing"]:
            return
        kind, idx = _parse_selection(str(obstacle_select.value))
        if kind != "cuboid":
            return
        env["cuboids"][idx]["dims"][0] = float(cuboid_length.value)
        _render_obstacle("cuboid", idx)

    @cuboid_width.on_update
    def _(_event) -> None:
        if _guard["syncing"]:
            return
        kind, idx = _parse_selection(str(obstacle_select.value))
        if kind != "cuboid":
            return
        env["cuboids"][idx]["dims"][1] = float(cuboid_width.value)
        _render_obstacle("cuboid", idx)

    @cuboid_height.on_update
    def _(_event) -> None:
        if _guard["syncing"]:
            return
        kind, idx = _parse_selection(str(obstacle_select.value))
        if kind != "cuboid":
            return
        env["cuboids"][idx]["dims"][2] = float(cuboid_height.value)
        _render_obstacle("cuboid", idx)

    @add_sphere_btn.on_click
    def _(_event) -> None:
        i = len(env["spheres"])
        env["spheres"].append({
            "name": f"sphere_{i}",
            "center": [0.3 + 0.08 * i, 0.0, 0.5],
            "radius": 0.08,
        })
        _render_obstacle("sphere", i)
        obstacle_select.options = tuple(_obstacle_options())
        obstacle_select.value = f"sphere::sphere_{i}"
        _sync_controls_from_selection()

    @add_cuboid_btn.on_click
    def _(_event) -> None:
        i = len(env["cuboids"])
        env["cuboids"].append({
            "name": f"cuboid_{i}",
            "center": [0.45 + 0.1 * i, 0.0, 0.2],
            "dims": [0.25, 0.25, 0.25],
            "wxyz": [1.0, 0.0, 0.0, 0.0],
        })
        _render_obstacle("cuboid", i)
        obstacle_select.options = tuple(_obstacle_options())
        obstacle_select.value = f"cuboid::cuboid_{i}"
        _sync_controls_from_selection()

    @save_btn.on_click
    def _(_event) -> None:
        args.env.write_text(json.dumps(env, indent=2), encoding="utf-8")
        print(f"Saved updated environment to {args.env}")

    _sync_controls_from_selection()

    print(
        f"Rendered {len(env.get('spheres', []))} spheres and "
        f"{len(env.get('cuboids', []))} cuboids. Press Ctrl+C to exit."
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
