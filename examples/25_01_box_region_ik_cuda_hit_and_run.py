"""Sample Panda IK configurations with end-effector coverage inside a box region using hit-and-run sampling.

Prerequisite:
    bash src/pyronot/cuda_kernels/build_hit_and_run_ik_cuda.sh
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyronot as pk
import viser
from pyronot.optimization_engines import hit_and_run_sample_box_region_cuda
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def _box_corners_edges(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    corners = np.array(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    return corners[edges]


def _stats_markdown(
    n_samples: int,
    requested_samples: int,
    solve_ms: float,
    inside_ratio: float,
    err_mean: float,
    final_entropy: float,
    max_entropy: float,
) -> str:
    count_str = (
        f"{n_samples} / {requested_samples} (partial)"
        if n_samples < requested_samples
        else f"{n_samples}"
    )
    return (
        f"Solved {count_str} samples in {solve_ms:.1f} ms\n\n"
        f"Inside-box ratio: {inside_ratio * 100.0:.2f}%\n\n"
        f"Mean final weighted error: {err_mean:.6f}\n\n"
        f"EE entropy: {final_entropy:.3f} / {max_entropy:.3f} nats"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seeds-per-launch", type=int, default=2048)
    parser.add_argument("--restarts-per-target", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=128,
        help="CUDA threads per block (multiple of 32; for current build use <= 384).",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help=(
            "Stop collecting samples once the Shannon entropy of the EE-point "
            "distribution reaches this value (nats). Max is log(10^3) ≈ 6.91 "
            "for the default 10-bin histogram. When omitted, collect --samples."
        ),
    )
    parser.add_argument("--entropy-bins", type=int, default=10)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch timing to diagnose loop overhead.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    max_tpb_by_smem = 384
    if (
        args.threads_per_block < 32
        or args.threads_per_block > 1024
        or args.threads_per_block % 32 != 0
    ):
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024].")
    if args.threads_per_block > max_tpb_by_smem:
        raise ValueError(
            f"threads_per_block={args.threads_per_block} exceeds the shared-memory limit for this build; "
            f"use <= {max_tpb_by_smem}."
        )

    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    target_link_name = "panda_hand"
    target_link_index = robot.links.names.index(target_link_name)

    hand_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    fixed_joint_mask = jnp.array(
        [name in hand_joint_names for name in robot.joints.actuated_names],
        dtype=jnp.int32,
    )

    previous_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2.0

    default_box_center = np.array([0.55, 0.0, 0.42], dtype=np.float32)
    default_box_dims = np.array([0.22, 0.34, 0.24], dtype=np.float32)
    box_center = default_box_center.copy()
    box_dims = default_box_dims.copy()

    call_kwargs = dict(
        robot=robot,
        target_link_index=target_link_index,
        previous_cfg=previous_cfg,
        num_samples=args.samples,
        seeds_per_launch=args.seeds_per_launch,
        restarts_per_target=args.restarts_per_target,
        max_iter=args.max_iter,
        n_iterations=args.n_iterations,
        noise_std=args.noise_std,
        threads_per_block=args.threads_per_block,
        fixed_joint_mask=fixed_joint_mask,
        memory_limit_gb=2.0,
        target_entropy=args.target_entropy,
        entropy_bins=args.entropy_bins,
        verbose=args.verbose,
    )

    from pyronot.optimization_engines._region_ik import _box_entropy

    max_entropy = float(np.log(args.entropy_bins**3))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/panda")
    center_x = server.gui.add_slider(
        "Box Center X",
        min=0.15,
        max=0.85,
        step=0.01,
        initial_value=float(box_center[0]),
    )
    center_y = server.gui.add_slider(
        "Box Center Y", min=-0.6, max=0.6, step=0.01, initial_value=float(box_center[1])
    )
    center_z = server.gui.add_slider(
        "Box Center Z", min=0.05, max=0.9, step=0.01, initial_value=float(box_center[2])
    )
    size_x = server.gui.add_slider(
        "Box Size X", min=0.02, max=0.8, step=0.01, initial_value=float(box_dims[0])
    )
    size_y = server.gui.add_slider(
        "Box Size Y", min=0.02, max=0.8, step=0.01, initial_value=float(box_dims[1])
    )
    size_z = server.gui.add_slider(
        "Box Size Z", min=0.02, max=0.8, step=0.01, initial_value=float(box_dims[2])
    )
    auto_resolve = server.gui.add_checkbox(
        "Auto Resolve On Box Change", initial_value=False
    )
    resolve_btn = server.gui.add_button("Resolve Samples")
    status_md = server.gui.add_markdown("Preparing solver...")
    idx_slider = server.gui.add_slider(
        "Sample Index", min=0, max=0, step=1, initial_value=0
    )
    play = server.gui.add_checkbox("Play", initial_value=True)

    solve_nonce = 0
    needs_solve = True
    cfgs_np = np.zeros((1, robot.joints.num_actuated_joints), dtype=np.float32)
    ee_np = np.zeros((1, 3), dtype=np.float32)
    target_np = np.zeros((1, 3), dtype=np.float32)

    def _gui_box() -> tuple[np.ndarray, np.ndarray]:
        center = np.array(
            [center_x.value, center_y.value, center_z.value], dtype=np.float32
        )
        dims = np.array([size_x.value, size_y.value, size_z.value], dtype=np.float32)
        box_min_np = center - 0.5 * dims
        box_max_np = center + 0.5 * dims
        return box_min_np, box_max_np

    def _set_status(text: str) -> None:
        if hasattr(status_md, "content"):
            status_md.content = text
        elif hasattr(status_md, "markdown"):
            status_md.markdown = text
        else:
            status_md.value = text

    def _draw_box() -> None:
        box_min_np, box_max_np = _gui_box()
        server.scene.add_line_segments(
            "/region/box",
            points=_box_corners_edges(box_min_np, box_max_np),
            colors=np.array([255, 182, 193], dtype=np.uint8),
            line_width=2.0,
        )

    def _solve_now() -> None:
        nonlocal solve_nonce, cfgs_np, ee_np, target_np
        box_min_np, box_max_np = _gui_box()
        box_min_jax = jnp.asarray(box_min_np, dtype=jnp.float32)
        box_max_jax = jnp.asarray(box_max_np, dtype=jnp.float32)
        _set_status("Solving...")

        if solve_nonce == 0:
            rng_key_warmup = jax.random.PRNGKey(0)
            t0 = time.perf_counter()
            warm_cfgs, _, _, _ = hit_and_run_sample_box_region_cuda(
                rng_key=rng_key_warmup,
                box_min=box_min_jax,
                box_max=box_max_jax,
                **call_kwargs,
            )
            warm_cfgs.block_until_ready()
            warmup_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Warmup (JIT compile + run): {warmup_ms:.1f} ms")

        solve_nonce += 1
        rng_key_run = jax.random.PRNGKey(solve_nonce)
        t0 = time.perf_counter()
        cfgs, ee_points, target_points, errors = hit_and_run_sample_box_region_cuda(
            rng_key=rng_key_run, box_min=box_min_jax, box_max=box_max_jax, **call_kwargs
        )
        cfgs.block_until_ready()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Solve {solve_nonce}: {solve_ms:.1f} ms")

        cfgs_np = np.asarray(cfgs)
        ee_np = np.asarray(ee_points)
        target_np = np.asarray(target_points)
        err_np = np.asarray(errors)

        inside_mask = np.all((ee_np >= box_min_np) & (ee_np <= box_max_np), axis=1)
        inside_ratio = float(np.mean(inside_mask)) if inside_mask.size else 0.0
        final_entropy = _box_entropy(ee_np, box_min_np, box_max_np, args.entropy_bins)

        rng = np.random.default_rng(solve_nonce)
        target_colors = rng.integers(
            0, 256, size=(target_np.shape[0], 3), dtype=np.uint8
        )
        ee_colors = rng.integers(0, 256, size=(ee_np.shape[0], 3), dtype=np.uint8)

        server.scene.add_point_cloud(
            "/region/target_points",
            points=target_np,
            colors=target_colors,
            point_size=0.003,
            point_shape="sparkle",
        )
        server.scene.add_point_cloud(
            "/region/ee_points",
            points=ee_np,
            colors=ee_colors,
            point_size=0.003,
            point_shape="circle",
        )

        idx_slider.max = max(cfgs_np.shape[0] - 1, 0)
        idx_slider.value = 0
        urdf_vis.update_cfg(cfgs_np[0])
        _set_status(
            _stats_markdown(
                n_samples=cfgs_np.shape[0],
                requested_samples=args.samples,
                solve_ms=solve_ms,
                inside_ratio=inside_ratio,
                err_mean=float(err_np.mean()) if err_np.size else 0.0,
                final_entropy=final_entropy,
                max_entropy=max_entropy,
            )
        )

    def _request_solve(_: object | None = None) -> None:
        nonlocal needs_solve
        needs_solve = True

    resolve_btn.on_click(_request_solve)

    last_box = np.concatenate([box_center, box_dims], axis=0)
    _draw_box()

    print(f"Viewer running at http://{args.host}:{args.port}")

    while True:
        current_box = np.array(
            [
                center_x.value,
                center_y.value,
                center_z.value,
                size_x.value,
                size_y.value,
                size_z.value,
            ],
            dtype=np.float32,
        )
        if not np.allclose(current_box, last_box):
            last_box = current_box
            _draw_box()
            if auto_resolve.value:
                needs_solve = True

        if needs_solve:
            try:
                _solve_now()
            except Exception as exc:
                _set_status(f"Solve failed: `{exc}`")
            needs_solve = False

        if play.value:
            idx_slider.value = (idx_slider.value + 1) % cfgs_np.shape[0]
        urdf_vis.update_cfg(cfgs_np[idx_slider.value])
        time.sleep(0.05)


if __name__ == "__main__":
    main()
