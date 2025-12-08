"""Visualize Box and its six HalfSpace faces in viser.

Run this example and use the transform controls to move/rotate the box
and the GUI numbers to change length/width/height interactively.
"""

import time

import numpy as np
import pyronot as pk
import viser
import trimesh
from pyronot.collision._geometry import Box, Sphere, Capsule
from pyronot.collision import collide


def main():
    """Start viser and visualize a Box and its HalfSpace faces."""

    # Initial box parameters
    center = np.array([0.0, 0.0, 0.5])
    length, width, height = 0.1, 0.1, 0.1

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    box_handle = server.scene.add_transform_controls(
        "/box_handle", scale=0.2, position=tuple(center), wxyz=(0, 0, 1, 0)
    )

    # Add transform controls for a sphere and a capsule
    sphere_handle = server.scene.add_transform_controls(
        "/sphere_handle", scale=0.15, position=(0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    sphere_radius_handle = server.gui.add_number("Sphere Radius", 0.05)

    cap_handle = server.scene.add_transform_controls(
        "/cap_handle", scale=0.2, position=(-0.3, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )
    cap_radius_handle = server.gui.add_number("Capsule Radius", 0.03)
    cap_height_handle = server.gui.add_number("Capsule Height", 0.2)

    length_handle = server.gui.add_number("Length", length)
    width_handle = server.gui.add_number("Width", width)
    height_handle = server.gui.add_number("Height", height)

    server.scene.add_mesh_trimesh(
        "/box/mesh", mesh=Box.from_center_and_dimensions(center, length, width, height).to_trimesh()
    )

    server.scene.add_mesh_trimesh("/box/polytope", mesh=trimesh.Trimesh())

    while True:
        pos = np.array(box_handle.position)
        wxyz = np.array(box_handle.wxyz)
        length = float(length_handle.value) if hasattr(length_handle, "value") else float(length_handle)
        width = float(width_handle.value) if hasattr(width_handle, "value") else float(width_handle)
        height = float(height_handle.value) if hasattr(height_handle, "value") else float(height_handle)

        box = Box.from_center_and_dimensions(center=pos, length=length, width=width, height=height, wxyz=wxyz)

        # Sphere
        sph_pos = np.array(sphere_handle.position)
        sph_wxyz = np.array(sphere_handle.wxyz)
        sph_rad = float(sphere_radius_handle.value) if hasattr(sphere_radius_handle, "value") else float(sphere_radius_handle)
        sphere = Sphere.from_center_and_radius(center=sph_pos, radius=sph_rad)

        # Capsule
        cap_pos = np.array(cap_handle.position)
        cap_wxyz = np.array(cap_handle.wxyz)
        cap_rad = float(cap_radius_handle.value) if hasattr(cap_radius_handle, "value") else float(cap_radius_handle)
        cap_h = float(cap_height_handle.value) if hasattr(cap_height_handle, "value") else float(cap_height_handle)
        capsule = Capsule.from_radius_height(radius=cap_rad, height=cap_h, position=cap_pos, wxyz=cap_wxyz)

        server.scene.add_mesh_trimesh("/box/mesh", mesh=box.to_trimesh())
        server.scene.add_mesh_trimesh("/sphere/mesh", mesh=sphere.to_trimesh())
        server.scene.add_mesh_trimesh("/cap/mesh", mesh=capsule.to_trimesh())

        poly_mesh = box.to_trimesh()
        server.scene.add_mesh_trimesh("/box/polytope", mesh=poly_mesh)

        # Collision checks between all unique pairs
        pairs = [
            ("Box", box, "Sphere", sphere),
            ("Box", box, "Capsule", capsule),
            ("Sphere", sphere, "Capsule", capsule),
        ]
        for name1, g1, name2, g2 in pairs:
            try:
                d = collide(g1, g2)
                # d is a jax Array; convert to python float if scalar
                d_val = float(d)
                if d_val < 0.0:
                    print(f"Collision detected {name1} vs {name2}: distance={d_val:.6f}")
            except Exception as e:
                print(f"Error computing collision {name1} vs {name2}: {e}")

        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
