"""Interactive IK in benchmark collision environment.

Move a target transform in viser and inspect live metrics:
- Position / rotation task error.
- Collision penalty and minimum signed distance.
- Collision-free flag (min signed distance > 0).
- Approximate total objective cost.

Usage examples:
    python examples/04_04_ik_with_coll_bench_env.py
    python examples/04_04_ik_with_coll_bench_env.py --robot fetch --solver ls-cuda
    python examples/04_04_ik_with_coll_bench_env.py --env resources/bench_env.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import viser
import yourdfpy

from pyroffi.collision import Box, RobotCollision, Sphere, collide
from pyroffi._robot_srdf_parser import read_disabled_collisions_from_srdf
from pyroffi.optimization_engines._hjcd_ik import hjcd_solve_cuda
from pyroffi.optimization_engines._ls_ik import ls_ik_solve, ls_ik_solve_cuda
from pyroffi.optimization_engines._mppi_ik import mppi_ik_solve_cuda
from pyroffi.optimization_engines._sqp_ik import sqp_ik_solve_cuda
from viser.extras import ViserUrdf


RESOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "resources"

ROBOT_URDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda_spherized.urdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch_spherized.urdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter_spherized.urdf",
    "g1": RESOURCE_ROOT / "g1_description" / "g1_29dof_with_hand_rev_1_0_spherized.urdf",
}

ROBOT_SRDFS = {
    "panda": RESOURCE_ROOT / "panda" / "panda.srdf",
    "fetch": RESOURCE_ROOT / "fetch" / "fetch.srdf",
    "baxter": RESOURCE_ROOT / "baxter" / "baxter.srdf",
    "g1": RESOURCE_ROOT / "g1_description" / "g1_29dof.srdf",
}

ROBOT_TARGET_LINK_CANDIDATES = {
    "panda": ("panda_hand",),
    "fetch": ("gripper_link",),
    "baxter": ("right_hand",),
    "g1": ("right_hand_palm_link", "left_hand_palm_link"),
}

ROBOT_FIXED_JOINT_NAMES = {
    "panda": ("panda_finger_joint1", "panda_finger_joint2"),
    "fetch": (),
    "baxter": (),
    "g1": (),
}

SOLVER_CHOICES = ("ls-jax", "ls-cuda", "hjcd-cuda", "sqp-cuda", "mppi-cuda")

IK_KWARGS = {
    "ls-jax": dict(num_seeds=32, max_iter=60, pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3),
    "ls-cuda": dict(
        num_seeds=128,
        max_iter=60,
        pos_weight=50.0,
        ori_weight=10.0,
        lambda_init=5e-3,
        eps_pos=1e-8,
        eps_ori=1e-8,
    ),
    "hjcd-cuda": dict(
        num_seeds=256,
        coarse_max_iter=20,
        lm_max_iter=40,
        lambda_init=1e-3,
        limit_prior_weight=1e-4,
        kick_scale=0.02,
    ),
    "sqp-cuda": dict(
        num_seeds=128,
        max_iter=60,
        n_inner_iters=2,
        pos_weight=50.0,
        ori_weight=10.0,
        lambda_init=5e-3,
        eps_pos=1e-8,
        eps_ori=1e-8,
    ),
    "mppi-cuda": dict(
        num_seeds=128,
        n_particles=16,
        n_mppi_iters=5,
        n_lbfgs_iters=25,
        m_lbfgs=5,
        pos_weight=50.0,
        ori_weight=10.0,
        sigma=0.3,
        mppi_temperature=0.05,
        eps_pos=1e-8,
        eps_ori=1e-8,
    ),
}

COLL_WEIGHT = 1e8
COLL_EPS = 0.005


def _resolve_target_link_name(robot_name: str, robot: pk.Robot) -> str:
    candidates = ROBOT_TARGET_LINK_CANDIDATES.get(robot_name, ())
    for name in candidates:
        if name in robot.links.names:
            return name
    raise ValueError(f"No valid target link found for robot '{robot_name}'. Tried {list(candidates)}")


def _default_env_file() -> pathlib.Path:
    return RESOURCE_ROOT / "bench_env.json"


def _default_srdf_for_robot(robot_name: str) -> pathlib.Path | None:
    """Resolve SRDF path for a robot, preferring explicit mapping then folder scan."""
    mapped = ROBOT_SRDFS.get(robot_name)
    if mapped is not None and mapped.exists():
        return mapped

    urdf_path = ROBOT_URDFS.get(robot_name)
    if urdf_path is None:
        return None

    srdf_candidates = sorted(urdf_path.parent.glob("*.srdf"))
    if len(srdf_candidates) == 1:
        return srdf_candidates[0]

    urdf_stem = urdf_path.stem
    for candidate in srdf_candidates:
        if urdf_stem.startswith(candidate.stem):
            return candidate

    return srdf_candidates[0] if srdf_candidates else None


def _disabled_pairs_from_srdf(srdf_path: pathlib.Path | None) -> tuple[tuple[str, str], ...]:
    if srdf_path is None or not srdf_path.exists():
        return ()
    try:
        pairs = read_disabled_collisions_from_srdf(srdf_path.as_posix())
        return tuple(
            (str(p["link1"]), str(p["link2"]))
            for p in pairs
            if p.get("link1") and p.get("link2")
        )
    except Exception as exc:
        print(f"Warning: failed to parse SRDF {srdf_path}: {exc}")
        return ()


def _load_env(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    return json.loads(path.read_text())


def _validate_env_dict(env: dict, path: pathlib.Path) -> None:
    if not isinstance(env, dict):
        raise ValueError(f"Environment JSON at {path} must be an object/dict")

    spheres = env.get("spheres", [])
    cuboids = env.get("cuboids", [])
    if not isinstance(spheres, list) or not isinstance(cuboids, list):
        raise ValueError(f"Environment JSON at {path} keys spheres/cuboids must be lists")


def _env_to_geoms(env: dict):
    obs_geoms: list = []
    for s in env.get("spheres", []):
        obs_geoms.append(
            Sphere.from_center_and_radius(
                np.array(s["center"], dtype=np.float32),
                np.array([s["radius"]], dtype=np.float32),
            )
        )
    for b in env.get("cuboids", []):
        d = b["dims"]
        obs_geoms.append(
            Box.from_center_and_dimensions(
                np.array(b["center"], dtype=np.float32),
                float(d[0]),
                float(d[1]),
                float(d[2]),
                wxyz=np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
            )
        )
    return obs_geoms


def _add_env_meshes(server: viser.ViserServer, env: dict) -> None:
    floor = env.get("floor", {})
    floor_pt = np.array(floor.get("point", [0.0, 0.0, 0.0]), dtype=np.float32)
    server.scene.add_grid(
        "/env/floor_grid",
        width=2.5,
        height=2.5,
        cell_size=0.1,
        position=(0.0, 0.0, float(floor_pt[2])),
    )

    for i, s in enumerate(env.get("spheres", [])):
        name = s.get("name", f"sphere_{i}")
        sphere = Sphere.from_center_and_radius(
            center=np.array(s["center"], dtype=np.float32),
            radius=float(s["radius"]),
        )
        server.scene.add_mesh_trimesh(f"/env/spheres/{name}", sphere.to_trimesh())

    for i, b in enumerate(env.get("cuboids", [])):
        name = b.get("name", f"cuboid_{i}")
        dims = np.array(b["dims"], dtype=np.float32)
        box = Box.from_center_and_dimensions(
            center=np.array(b["center"], dtype=np.float32),
            length=float(dims[0]),
            width=float(dims[1]),
            height=float(dims[2]),
            wxyz=np.array(b.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
        )
        server.scene.add_mesh_trimesh(f"/env/cuboids/{name}", box.to_trimesh())


def _solve_once(
    solver_name: str,
    robot: pk.Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
    rng_key: jax.Array,
    previous_cfg: jax.Array,
    fixed_joint_mask: jax.Array,
    collision_penalty_fn,
    dummy_constraint_arg,
) -> jax.Array:
    kwargs = IK_KWARGS[solver_name]

    if solver_name == "ls-jax":
        return ls_ik_solve(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            fixed_joint_mask=fixed_joint_mask,
            constraint_fns=(collision_penalty_fn,),
            constraint_args=(dummy_constraint_arg,),
            constraint_weights=jnp.array([COLL_WEIGHT]),
            **kwargs,
        )

    cuda_constraints = dict(
        constraints=[collision_penalty_fn],
        constraint_args=[dummy_constraint_arg],
        constraint_weights=[COLL_WEIGHT],
    )

    if solver_name == "ls-cuda":
        return ls_ik_solve_cuda(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            fixed_joint_mask=fixed_joint_mask,
            **cuda_constraints,
            **kwargs,
        )
    if solver_name == "hjcd-cuda":
        return hjcd_solve_cuda(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            fixed_joint_mask=fixed_joint_mask,
            **cuda_constraints,
            **kwargs,
        )
    if solver_name == "sqp-cuda":
        return sqp_ik_solve_cuda(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            fixed_joint_mask=fixed_joint_mask,
            **cuda_constraints,
            **kwargs,
        )
    if solver_name == "mppi-cuda":
        return mppi_ik_solve_cuda(
            robot=robot,
            target_link_indices=(target_link_index,),
            target_poses=(target_pose,),
            rng_key=rng_key,
            previous_cfg=previous_cfg,
            fixed_joint_mask=fixed_joint_mask,
            **cuda_constraints,
            **kwargs,
        )

    raise ValueError(f"Unknown solver: {solver_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive collision-aware IK in benchmark environment.")
    parser.add_argument("--robot", choices=tuple(ROBOT_URDFS.keys()), default="panda")
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="ls-jax")
    parser.add_argument(
        "--env",
        type=pathlib.Path,
        default=None,
        help="Path to environment JSON (default: resources/bench_env.json).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--disable_srdf",
        action="store_true",
        help="Ignore SRDF disabled-collision pairs even if an SRDF exists.",
    )
    args = parser.parse_args()

    urdf_path = ROBOT_URDFS[args.robot]
    if not urdf_path.exists():
        raise FileNotFoundError(f"Spherized URDF not found: {urdf_path}")

    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf)
    srdf_path = _default_srdf_for_robot(args.robot)
    ignore_pairs = () if args.disable_srdf else _disabled_pairs_from_srdf(srdf_path)
    robot_coll = RobotCollision.from_urdf(urdf, user_ignore_pairs=ignore_pairs)

    target_link_name = _resolve_target_link_name(args.robot, robot)
    target_link_index = robot.links.names.index(target_link_name)

    fixed_joint_names = ROBOT_FIXED_JOINT_NAMES.get(args.robot, ())
    fixed_joint_mask = jnp.array(
        [name in fixed_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.bool_,
    )

    env_path = args.env if args.env is not None else _default_env_file()
    env = _load_env(env_path)
    _validate_env_dict(env, env_path)

    obs_geoms = _env_to_geoms(env)
    coll_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)

    lo = jnp.array(robot.joints.lower_limits, dtype=jnp.float32)
    hi = jnp.array(robot.joints.upper_limits, dtype=jnp.float32)

    def collision_penalty_fn(cfg, robot_arg, _dummy):
        coll_geom = robot_coll.at_config(robot_arg, cfg)
        penalty = jnp.zeros(())
        for obs in obs_geoms:
            d_obs = coll_vs_world(coll_geom, obs.broadcast_to((1,)))
            penalty = penalty + jnp.sum(jax.nn.softplus(-d_obs / COLL_EPS) * COLL_EPS)
        return penalty

    @jax.jit
    def min_signed_dist_fn(cfg):
        coll_geom = robot_coll.at_config(robot, cfg)
        if len(obs_geoms) == 0:
            return jnp.inf
        dists = []
        for obs in obs_geoms:
            dists.append(jnp.min(coll_vs_world(coll_geom, obs.broadcast_to((1,)))))
        return jnp.min(jnp.stack(dists))

    server = viser.ViserServer(host=args.host, port=args.port)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    _add_env_meshes(server, env)

    target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=(0.5, 0.0, 0.5),
        wxyz=(0.0, 0.0, 1.0, 0.0),
    )

    with server.gui.add_folder("IK Diagnostics"):
        solver_label = server.gui.add_text("Solver", initial_value=args.solver, disabled=True)
        link_label = server.gui.add_text("Target Link", initial_value=target_link_name, disabled=True)
        elapsed_ms_h = server.gui.add_number("Elapsed (ms)", initial_value=0.0, step=1e-6, disabled=True)
        pos_err_h = server.gui.add_number("Position error (mm)", initial_value=0.0, step=1e-6, disabled=True)
        rot_err_h = server.gui.add_number("Rotation error (rad)", initial_value=0.0, step=1e-6, disabled=True)
        min_dist_h = server.gui.add_number("Min signed distance (mm)", initial_value=0.0, step=1e-6, disabled=True)
        coll_free_h = server.gui.add_number("Collision-free (1/0)", initial_value=0.0, step=1.0, disabled=True)
        coll_pen_h = server.gui.add_number("Collision penalty", initial_value=0.0, step=1e-9, disabled=True)
        task_cost_h = server.gui.add_number("Task cost", initial_value=0.0, step=1e-9, disabled=True)
        weighted_coll_cost_h = server.gui.add_number("Weighted collision cost", initial_value=0.0, step=1e-3, disabled=True)
        total_cost_h = server.gui.add_number("Approx total cost", initial_value=0.0, step=1e-3, disabled=True)

    print(f"Viser server started at http://{args.host}:{args.port}")
    print(f"Robot={args.robot}  solver={args.solver}  target_link={target_link_name}")
    print(f"Environment={env_path}")
    if not args.disable_srdf and srdf_path is not None and srdf_path.exists():
        print(f"SRDF={srdf_path} ({len(ignore_pairs)} disabled pairs)")
    else:
        print("SRDF disabled pairs: none")

    rng_key = jax.random.PRNGKey(0)
    solution = (lo + hi) / 2.0
    dummy_constraint_arg = jnp.zeros(())

    pos_weight = IK_KWARGS[args.solver].get("pos_weight", 50.0)
    ori_weight = IK_KWARGS[args.solver].get("ori_weight", 10.0)

    while True:
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(wxyz=jnp.array(target_handle.wxyz)),
            translation=jnp.array(target_handle.position),
        )

        rng_key, subkey = jax.random.split(rng_key)
        t0 = time.perf_counter()
        solution = _solve_once(
            solver_name=args.solver,
            robot=robot,
            target_link_index=target_link_index,
            target_pose=target_pose,
            rng_key=subkey,
            previous_cfg=solution,
            fixed_joint_mask=fixed_joint_mask,
            collision_penalty_fn=collision_penalty_fn,
            dummy_constraint_arg=dummy_constraint_arg,
        )
        solution.block_until_ready()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        link_poses = robot.forward_kinematics(solution)
        actual_pose = jaxlie.SE3(link_poses[target_link_index])
        pos_err = jnp.linalg.norm(actual_pose.translation() - target_pose.translation())
        rot_err = jnp.linalg.norm((target_pose.rotation().inverse() @ actual_pose.rotation()).log())

        coll_pen = collision_penalty_fn(solution, robot, dummy_constraint_arg)
        min_dist = min_signed_dist_fn(solution)
        task_cost = pos_weight * (pos_err ** 2) + ori_weight * (rot_err ** 2)
        weighted_coll_cost = COLL_WEIGHT * coll_pen
        total_cost = task_cost + weighted_coll_cost

        urdf_vis.update_cfg(np.array(solution))

        elapsed_ms_h.value = float(elapsed_ms)
        pos_err_h.value = float(pos_err) * 1000.0
        rot_err_h.value = float(rot_err)
        min_dist_h.value = float(min_dist) * 1000.0
        coll_free_h.value = 1.0 if float(min_dist) > 0.0 else 0.0
        coll_pen_h.value = float(coll_pen)
        task_cost_h.value = float(task_cost)
        weighted_coll_cost_h.value = float(weighted_coll_cost)
        total_cost_h.value = float(total_cost)


if __name__ == "__main__":
    main()