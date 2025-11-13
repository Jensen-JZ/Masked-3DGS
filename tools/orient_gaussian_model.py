#!/usr/bin/env python3
"""Rotate a trained Gaussian Splat model to a canonical vehicle frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from reorient_colmap import (  # type: ignore
    compute_alignment,
    read_images_binary,
    read_points3d_binary,
)


def _parse_alignment_matrix(path: Path) -> np.ndarray:
    values: List[List[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append([float(v) for v in line.split()])
    if len(values) != 3 or any(len(row) != 3 for row in values):
        raise ValueError(f"Alignment matrix at {path} must contain three whitespace-separated rows")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Alignment matrix must be 3x3")
    return matrix


def _load_or_compute_alignment(matrix_path: Path | None, sparse_dir: Path | None) -> np.ndarray:
    if matrix_path is not None:
        matrix = _parse_alignment_matrix(matrix_path)
    elif sparse_dir is not None:
        images = read_images_binary(sparse_dir / "images.bin")
        points = read_points3d_binary(sparse_dir / "points3D.bin")
        matrix = compute_alignment(points, images)
    else:
        raise ValueError("Provide either --alignment-matrix or --sparse to determine the rotation")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5):
        raise ValueError("Alignment matrix is not orthonormal")
    if np.linalg.det(matrix) < 0:
        raise ValueError("Alignment matrix must have positive determinant")
    return matrix.astype(np.float64)


def _load_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    header_lines: List[str] = []
    property_names: List[str] = []
    vertex_count = None

    with path.open("rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise EOFError("Unexpected end of PLY header")
            line = raw.decode("ascii").strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property"):
                tokens = line.split()
                if len(tokens) != 3 or tokens[1] != "float":
                    raise ValueError("Only float properties are supported")
                property_names.append(tokens[2])
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError("PLY file is missing vertex count")

        dtype = np.dtype([(name, "<f4") for name in property_names])
        data = np.fromfile(fh, dtype=dtype, count=vertex_count)

    prop_set = set(property_names)

    coords = np.stack([data[axis] for axis in ("x", "y", "z")], axis=1)
    normals = (
        np.stack([data[field] for field in ("nx", "ny", "nz")], axis=1)
        if {"nx", "ny", "nz"}.issubset(prop_set)
        else None
    )
    rotations = (
        np.stack([data[f"rot_{i}"] for i in range(4)], axis=1)
        if {f"rot_{i}" for i in range(4)}.issubset(prop_set)
        else None
    )
    scalars = data
    return coords.astype(np.float64), normals, rotations, scalars, header_lines


def _write_ply(path: Path, header_lines: Sequence[str], scalars: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for line in header_lines:
            fh.write((line + "\n").encode("ascii"))
        scalars.tofile(fh)


def _quats_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_quats(rotations: np.ndarray) -> np.ndarray:
    traces = np.trace(rotations, axis1=1, axis2=2)
    quats = np.empty((rotations.shape[0], 4), dtype=np.float64)

    positive = traces > 0.0
    if np.any(positive):
        t = traces[positive]
        s = np.sqrt(t + 1.0) * 2.0
        quats[positive, 0] = 0.25 * s
        quats[positive, 1] = (rotations[positive, 2, 1] - rotations[positive, 1, 2]) / s
        quats[positive, 2] = (rotations[positive, 0, 2] - rotations[positive, 2, 0]) / s
        quats[positive, 3] = (rotations[positive, 1, 0] - rotations[positive, 0, 1]) / s

    negative = ~positive
    if np.any(negative):
        Rn = rotations[negative]
        cond1 = (Rn[:, 0, 0] > Rn[:, 1, 1]) & (Rn[:, 0, 0] > Rn[:, 2, 2])
        cond2 = ~cond1 & (Rn[:, 1, 1] > Rn[:, 2, 2])
        cond3 = ~(cond1 | cond2)

        if np.any(cond1):
            Rc = Rn[cond1]
            s = np.sqrt(1.0 + Rc[:, 0, 0] - Rc[:, 1, 1] - Rc[:, 2, 2]) * 2.0
            quats[negative][cond1, 0] = (Rc[:, 2, 1] - Rc[:, 1, 2]) / s
            quats[negative][cond1, 1] = 0.25 * s
            quats[negative][cond1, 2] = (Rc[:, 0, 1] + Rc[:, 1, 0]) / s
            quats[negative][cond1, 3] = (Rc[:, 0, 2] + Rc[:, 2, 0]) / s

        if np.any(cond2):
            Rc = Rn[cond2]
            s = np.sqrt(1.0 + Rc[:, 1, 1] - Rc[:, 0, 0] - Rc[:, 2, 2]) * 2.0
            quats[negative][cond2, 0] = (Rc[:, 0, 2] - Rc[:, 2, 0]) / s
            quats[negative][cond2, 1] = (Rc[:, 0, 1] + Rc[:, 1, 0]) / s
            quats[negative][cond2, 2] = 0.25 * s
            quats[negative][cond2, 3] = (Rc[:, 1, 2] + Rc[:, 2, 1]) / s

        if np.any(cond3):
            Rc = Rn[cond3]
            s = np.sqrt(1.0 + Rc[:, 2, 2] - Rc[:, 0, 0] - Rc[:, 1, 1]) * 2.0
            quats[negative][cond3, 0] = (Rc[:, 1, 0] - Rc[:, 0, 1]) / s
            quats[negative][cond3, 1] = (Rc[:, 0, 2] + Rc[:, 2, 0]) / s
            quats[negative][cond3, 2] = (Rc[:, 1, 2] + Rc[:, 2, 1]) / s
            quats[negative][cond3, 3] = 0.25 * s

    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    mask = quats[:, 0] < 0
    quats[mask] *= -1.0
    return quats


def _apply_rotation_to_ply(
    ply_path: Path,
    output_path: Path,
    rotation_matrix: np.ndarray,
    flip: Tuple[int, int, int],
) -> None:
    coords, normals, rotations, scalars, header_lines = _load_ply(ply_path)

    flip_matrix = np.diag(flip).astype(np.float64)
    transform = flip_matrix @ rotation_matrix

    rotated = (transform @ coords.T).T
    scalars["x"] = rotated[:, 0].astype(np.float32)
    scalars["y"] = rotated[:, 1].astype(np.float32)
    scalars["z"] = rotated[:, 2].astype(np.float32)

    if normals is not None:
        rotated_normals = (transform @ normals.T).T
        scalars["nx"] = rotated_normals[:, 0].astype(np.float32)
        scalars["ny"] = rotated_normals[:, 1].astype(np.float32)
        scalars["nz"] = rotated_normals[:, 2].astype(np.float32)

    if rotations is not None:
        local_rot = _quats_to_rotmat(rotations.astype(np.float64))
        combined = transform @ local_rot
        quats = _rotmat_to_quats(combined)
        for i in range(4):
            scalars[f"rot_{i}"] = quats[:, i].astype(np.float32)

    _write_ply(output_path, header_lines, scalars)


def _apply_rotation_to_cameras(
    cameras_path: Path, output_path: Path | None, rotation_matrix: np.ndarray, flip: Tuple[int, int, int]
) -> None:
    data = json.loads(cameras_path.read_text(encoding="utf-8"))
    flip_matrix = np.diag(flip).astype(np.float64)
    transform = flip_matrix @ rotation_matrix

    for entry in data:
        R_wc = np.asarray(entry["rotation"], dtype=np.float64)
        t_wc = np.asarray(entry["position"], dtype=np.float64)
        center = -R_wc.T @ t_wc
        center_rotated = transform @ center
        R_new = R_wc @ transform.T
        t_new = -R_new @ center_rotated
        entry["rotation"] = R_new.tolist()
        entry["position"] = t_new.tolist()

    target = output_path if output_path is not None else cameras_path
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_alignment(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ["# Rows describe new axes expressed in original coordinates."]
    for row in matrix:
        content.append(" ".join(f"{value:.9f}" for value in row))
    path.write_text("\n".join(content) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate Gaussian splats into a canonical vehicle frame")
    parser.add_argument("--point-cloud", type=Path, required=True, help="Path to point_cloud.ply produced by training")
    parser.add_argument("--output", type=Path, help="Destination PLY. Defaults to <point_cloud>_aligned.ply")
    parser.add_argument("--sparse", type=Path, help="COLMAP sparse/0 directory for estimating the rotation")
    parser.add_argument("--alignment-matrix", type=Path, help="Skip estimation and load an existing 3x3 rotation")
    parser.add_argument("--cameras", type=Path, help="Optional cameras.json to rotate together with the splats")
    parser.add_argument("--cameras-output", type=Path, help="Output path for rotated cameras.json")
    parser.add_argument("--save-matrix", type=Path, help="Where to store the applied rotation for record keeping")
    parser.add_argument("--flip-x", action="store_true", help="Flip the resulting X axis if the car points backwards")
    parser.add_argument("--flip-y", action="store_true", help="Flip the resulting Y axis")
    parser.add_argument("--flip-z", action="store_true", help="Flip the resulting Z axis")
    args = parser.parse_args()

    rotation = _load_or_compute_alignment(args.alignment_matrix, args.sparse)
    flips = (
        -1 if args.flip_x else 1,
        -1 if args.flip_y else 1,
        -1 if args.flip_z else 1,
    )

    output_path = (
        args.output
        if args.output is not None
        else args.point_cloud.with_name(args.point_cloud.stem + "_aligned.ply")
    )

    _apply_rotation_to_ply(args.point_cloud, output_path, rotation, flips)

    if args.cameras is not None:
        _apply_rotation_to_cameras(args.cameras, args.cameras_output, rotation, flips)

    if args.save_matrix is not None:
        _save_alignment(rotation, args.save_matrix)

    np.set_printoptions(precision=6, suppress=True)
    print("Applied rotation (rows = new axes in old coordinates):")
    print(rotation)
    if any(axis == -1 for axis in flips):
        print(f"Axis flips applied: {flips}")
    print(f"Rotated point cloud written to {output_path}")
    if args.cameras is not None:
        dest = args.cameras_output if args.cameras_output is not None else args.cameras
        print(f"Updated cameras written to {dest}")
    if args.save_matrix is not None:
        print(f"Saved rotation matrix to {args.save_matrix}")


if __name__ == "__main__":
    main()
