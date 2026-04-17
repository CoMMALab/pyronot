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
import viser

from pyroffi.collision import Box, Sphere


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bench_env.json in viser.")
    parser.add_argument(
        "--env",
        type=pathlib.Path,
        default=pathlib.Path("resources/bench_env.json"),
        help="Path to environment JSON.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    args = parser.parse_args()

    env = _load_env(args.env)
    env.setdefault("spheres", [])
    env.setdefault("cuboids", [])

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")
    print(f"Loaded env: {args.env}")

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
        server.scene.add_mesh_trimesh(f"/env/spheres/{name}", sphere.to_trimesh())

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
            center = np.array(s["center"], dtype=np.float32)
            radius = float(s["radius"])
            sphere = Sphere.from_center_and_radius(center=center, radius=radius)
            server.scene.add_mesh_trimesh(f"/env/spheres/{name}", sphere.to_trimesh())
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
