#!/usr/bin/env python3
"""Utilities for baking "shadow gaussians" into a trained 3D Gaussian Splatting asset.

This script loads an existing point_cloud.ply produced by the training pipeline,
generates a flat rounded-rectangle scaffold under the asset footprint, and appends
shadow-specific gaussians with dark DC color, zero higher-order SH, and tapered
opacity. The output PLY can be re-used by the renderer with the shadow baked in.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # Spherical harmonics constant for DC band
AXIS_FIELDS = ("x", "y", "z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append flattened shadow gaussians to a Masked-3DGS point cloud PLY"
    )
    parser.add_argument("input", type=Path, help="Source point_cloud.ply produced by training")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for the augmented PLY (may be identical to input with --overwrite)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=4000,
        help="Approximate number of shadow gaussians to generate (default: 4000)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Fractional padding added to the X/Z footprint before sampling (default: 0.05)",
    )
    parser.add_argument(
        "--corner-ratio",
        type=float,
        default=0.2,
        help="Corner radius as a fraction of the min half-extent (default: 0.2)",
    )
    parser.add_argument(
        "--ground-axis",
        type=str,
        choices=["auto", "x", "y", "z"],
        default="auto",
        help="Axis treated as 'up'. Auto picks the axis with the smallest extent (default: auto)",
    )
    parser.add_argument(
        "--max-opacity",
        type=float,
        default=0.7,
        help="Peak opacity at the shadow center before sigmoid inverse (default: 0.7)",
    )
    parser.add_argument(
        "--falloff-power",
        type=float,
        default=2.0,
        help="Power applied to the normalized radial distance for the opacity falloff (default: 2.0)",
    )
    parser.add_argument(
        "--falloff-gamma",
        type=float,
        default=1.0,
        help="Gamma applied after the power falloff to tweak softness (default: 1.0)",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=-0.01,
        help="Offset (along the up axis) applied relative to the asset's minimum value (default: -0.01)",
    )
    parser.add_argument(
        "--y-scale",
        type=float,
        default=1e-3,
        help="Actual (post-exp) scale along the up axis for shadow gaussians (default: 1e-3)",
    )
    parser.add_argument(
        "--xy-scale-multiplier",
        type=float,
        default=1.25,
        help="Multiplier applied to XY grid spacing to set Gaussian footprint (default: 1.25)",
    )
    parser.add_argument(
        "--min-xy-scale",
        type=float,
        default=5e-3,
        help="Lower bound on XY scales to avoid degenerate splats (default: 5e-3)",
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        default=(0.05, 0.05, 0.05),
        help="RGB color (0-1) for the shadow DC term (default: 0.05 0.05 0.05)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing directly to the input path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used when subsampling the scaffold (default: 2025)",
    )
    return parser.parse_args()


def _safe_logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    clamped = np.clip(x, eps, 1.0 - eps)
    return np.log(clamped / (1.0 - clamped))


def _rgb_to_sh_dc(rgb: Sequence[float]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.float64)
    if rgb_arr.shape != (3,):
        raise ValueError("Shadow color must be a sequence of three floats (RGB)")
    return (rgb_arr - 0.5) / C0


def _resolve_ground_axis(choice: str, mins: dict, maxs: dict) -> str:
    choice_lower = choice.lower()
    if choice_lower == "auto":
        ranges = {axis: maxs[axis] - mins[axis] for axis in AXIS_FIELDS}
        return min(ranges, key=ranges.get)
    if choice_lower not in AXIS_FIELDS:
        raise ValueError(f"Invalid ground axis '{choice}'. Expected one of auto/x/y/z")
    return choice_lower


def _rounded_rectangle_mask(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    half_x: float,
    half_z: float,
    corner_radius: float,
) -> np.ndarray:
    abs_x = np.abs(x_grid)
    abs_z = np.abs(z_grid)
    rect_x = np.maximum(abs_x - (half_x - corner_radius), 0.0)
    rect_z = np.maximum(abs_z - (half_z - corner_radius), 0.0)
    return (rect_x * rect_x + rect_z * rect_z) <= corner_radius * corner_radius + 1e-8


def _sample_shadow_scaffold(
    target_count: int,
    half_x: float,
    half_z: float,
    corner_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    attempt = 0
    grid_side = max(4, int(math.sqrt(target_count / 0.75)))
    while attempt < 6:
        x_coords = np.linspace(-half_x, half_x, grid_side, dtype=np.float64)
        z_coords = np.linspace(-half_z, half_z, grid_side, dtype=np.float64)
        grid_x, grid_z = np.meshgrid(x_coords, z_coords, indexing="xy")
        mask = _rounded_rectangle_mask(grid_x, grid_z, half_x, half_z, corner_radius)
        points = np.stack((grid_x[mask], grid_z[mask]), axis=1)
        if points.shape[0] >= target_count or attempt == 5:
            break
        grid_side += 2
        attempt += 1

    if points.size == 0:
        raise RuntimeError("Failed to generate shadow scaffold points; check footprint dimensions")

    if points.shape[0] > target_count:
        indices = rng.choice(points.shape[0], size=target_count, replace=False)
        points = points[indices]
    return points


def _prepare_shadow_vertices(
    dtype: np.dtype,
    plane_points: np.ndarray,
    plane_fields: Tuple[str, str],
    up_field: str,
    up_value: float,
    color_sh: np.ndarray,
    plane_scales: Tuple[float, float],
    up_scale: float,
    max_opacity: float,
    falloff_power: float,
    falloff_gamma: float,
) -> np.ndarray:
    shadow_vertices = np.zeros(plane_points.shape[0], dtype=dtype)

    center = plane_points.mean(axis=0, keepdims=True)
    deltas = plane_points - center
    radial = np.linalg.norm(deltas, axis=1)
    radial_max = np.max(radial)
    if radial_max < 1e-6:
        falloff = np.ones_like(radial)
    else:
        normalized = np.clip(radial / radial_max, 0.0, 1.0)
        falloff = (1.0 - normalized ** falloff_power) ** falloff_gamma

    plane_scale_u, plane_scale_v = plane_scales
    plane_scale_u = max(plane_scale_u, 1e-6)
    plane_scale_v = max(plane_scale_v, 1e-6)
    up_scale = max(up_scale, 1e-6)

    log_scales_u = np.log(np.full(plane_points.shape[0], plane_scale_u, dtype=np.float64))
    log_scales_v = np.log(np.full(plane_points.shape[0], plane_scale_v, dtype=np.float64))
    log_scales_up = np.log(np.full(plane_points.shape[0], up_scale, dtype=np.float64))

    opacity = max_opacity * falloff
    logit_opacity = _safe_logit(opacity)

    # Populate structured array fields
    axis_to_log_scale = {
        plane_fields[0]: log_scales_u.astype(np.float32),
        plane_fields[1]: log_scales_v.astype(np.float32),
        up_field: log_scales_up.astype(np.float32),
    }

    shadow_vertices[plane_fields[0]] = plane_points[:, 0].astype(np.float32)
    shadow_vertices[plane_fields[1]] = plane_points[:, 1].astype(np.float32)
    shadow_vertices[up_field] = np.full(plane_points.shape[0], up_value, dtype=np.float32)

    if "nx" in shadow_vertices.dtype.names:
        shadow_vertices["nx"] = 0.0
    if "ny" in shadow_vertices.dtype.names:
        shadow_vertices["ny"] = 0.0
    if "nz" in shadow_vertices.dtype.names:
        shadow_vertices["nz"] = 0.0

    dc_fields = sorted((name for name in shadow_vertices.dtype.names if name.startswith("f_dc_")), key=lambda n: int(n.split("_")[-1]))
    if len(dc_fields) < 3:
        raise RuntimeError("Expected at least three f_dc_* fields for RGB DC coefficients")
    for idx, field in enumerate(dc_fields[:3]):
        shadow_vertices[field] = color_sh[idx].astype(np.float32)

    for field in (name for name in shadow_vertices.dtype.names if name.startswith("f_rest_")):
        shadow_vertices[field] = 0.0

    shadow_vertices["opacity"] = logit_opacity.astype(np.float32)

    scale_fields = sorted((name for name in shadow_vertices.dtype.names if name.startswith("scale_")), key=lambda n: int(n.split("_")[-1]))
    if len(scale_fields) < 3:
        raise RuntimeError("Expected three scale_* fields for XYZ scales")
    for idx, axis in enumerate(AXIS_FIELDS):
        if axis not in axis_to_log_scale:
            raise RuntimeError(f"Missing log scale for axis '{axis}'")
        shadow_vertices[scale_fields[idx]] = axis_to_log_scale[axis]

    rot_fields = sorted((name for name in shadow_vertices.dtype.names if name.startswith("rot_")), key=lambda n: int(n.split("_")[-1]))
    if len(rot_fields) < 4:
        raise RuntimeError("Expected four rot_* fields for quaternion")
    shadow_vertices[rot_fields[0]] = 1.0  # w component of unit quaternion
    for field in rot_fields[1:]:
        shadow_vertices[field] = 0.0

    return shadow_vertices


def add_shadow_gaussians_to_ply(
    input_path: Path,
    output_path: Path,
    *,
    target_count: int = 4000,
    margin: float = 0.05,
    corner_ratio: float = 0.2,
    ground_axis: str = "auto",
    max_opacity: float = 0.7,
    falloff_power: float = 2.0,
    falloff_gamma: float = 1.0,
    y_offset: float = -0.01,
    y_scale: float = 1e-3,
    xy_scale_multiplier: float = 1.25,
    min_xy_scale: float = 5e-3,
    color: Sequence[float] = (0.05, 0.05, 0.05),
    overwrite: bool = False,
    seed: int = 2025,
    verbose: bool = False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve() and not overwrite:
        raise ValueError("Use --overwrite to replace the input file in-place")

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    if target_count <= 0:
        raise ValueError("target_count must be positive")

    if not 0.0 < max_opacity < 1.0:
        raise ValueError("max_opacity must be in (0, 1)")

    if y_scale <= 0.0:
        raise ValueError("y_scale must be positive")

    color_sh = _rgb_to_sh_dc(color)

    with input_path.open("rb") as handle:
        ply_data = PlyData.read(handle)
    vertex_data = np.asarray(ply_data["vertex"].data).copy()

    mins = {axis: float(np.min(vertex_data[axis])) for axis in AXIS_FIELDS}
    maxs = {axis: float(np.max(vertex_data[axis])) for axis in AXIS_FIELDS}

    up_field = _resolve_ground_axis(ground_axis, mins, maxs)
    plane_fields = tuple(axis for axis in AXIS_FIELDS if axis != up_field)

    plane_center = np.array([0.5 * (mins[field] + maxs[field]) for field in plane_fields], dtype=np.float64)
    half_extents = []
    for field in plane_fields:
        extent = (maxs[field] - mins[field]) * (1.0 + 2.0 * margin)
        half_extents.append(max(extent * 0.5, 1e-5))
    half_u, half_v = half_extents

    corner_radius_val = np.clip(corner_ratio, 0.0, 0.499) * min(half_u, half_v)
    rng = np.random.default_rng(seed)

    scaffold_local = _sample_shadow_scaffold(target_count, half_u, half_v, corner_radius_val, rng)
    scaffold_world = scaffold_local + plane_center

    if scaffold_world.shape[0] < target_count:
        print(
            f"[shadow-gaussians] Warning: generated {scaffold_world.shape[0]} points (< target {target_count}).",
            "Consider reducing target_count or adjusting corner_ratio/margin.",
        )

    if scaffold_world.shape[0] == 0:
        raise RuntimeError("Shadow scaffold is empty after sampling")

    unique_u = np.unique(np.round(scaffold_local[:, 0], decimals=6))
    unique_v = np.unique(np.round(scaffold_local[:, 1], decimals=6))
    approx_spacing_u = (unique_u.ptp() or (2.0 * half_u)) / max(unique_u.size - 1, 1)
    approx_spacing_v = (unique_v.ptp() or (2.0 * half_v)) / max(unique_v.size - 1, 1)

    base_scale_u = max(min_xy_scale, approx_spacing_u * xy_scale_multiplier)
    base_scale_v = max(min_xy_scale, approx_spacing_v * xy_scale_multiplier)

    up_value = mins[up_field] + y_offset

    shadow_vertices = _prepare_shadow_vertices(
        vertex_data.dtype,
        scaffold_world,
        plane_fields,
        up_field,
        up_value,
        color_sh,
        (base_scale_u, base_scale_v),
        y_scale,
        max_opacity,
        falloff_power,
        falloff_gamma,
    )

    combined_vertices = np.concatenate([vertex_data, shadow_vertices])
    vertex_element = PlyElement.describe(combined_vertices, "vertex")

    ply_out = PlyData([vertex_element], text=ply_data.text, byte_order=ply_data.byte_order)
    for comment in ply_data.comments:
        ply_out.comments.append(comment)
    for obj_info in ply_data.obj_info:
        ply_out.obj_info.append(obj_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        ply_out.write(handle)

    if verbose:
        print(
            f"[shadow-gaussians] Appended {shadow_vertices.shape[0]} shadow points;"
            f" total vertices: {combined_vertices.shape[0]}"
        )

    return shadow_vertices.shape[0], combined_vertices.shape[0]


def add_shadow_gaussians(args: argparse.Namespace) -> None:
    shadow_count, total = add_shadow_gaussians_to_ply(
        args.input,
        args.output,
        target_count=args.target_count,
        margin=args.margin,
        corner_ratio=args.corner_ratio,
    ground_axis=args.ground_axis,
        max_opacity=args.max_opacity,
        falloff_power=args.falloff_power,
        falloff_gamma=args.falloff_gamma,
        y_offset=args.y_offset,
        y_scale=args.y_scale,
        xy_scale_multiplier=args.xy_scale_multiplier,
        min_xy_scale=args.min_xy_scale,
        color=args.color,
        overwrite=args.overwrite,
        seed=args.seed,
        verbose=False,
    )
    print(
        f"[shadow-gaussians] Done. Appended {shadow_count} shadow points;"
        f" total vertices: {total}"
    )


if __name__ == "__main__":
    add_shadow_gaussians(_parse_args())
