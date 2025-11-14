#!/usr/bin/env python3
"""Rotate a COLMAP sparse model so that up aligns with +Z and forward aligns with +X."""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    def rotation_matrix(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def copy_with(self, *, qvec: np.ndarray, tvec: np.ndarray) -> "Image":
        return Image(
            id=self.id,
            qvec=qvec,
            tvec=tvec,
            camera_id=self.camera_id,
            name=self.name,
            xys=self.xys,
            point3D_ids=self.point3D_ids,
        )


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


CAMERA_MODELS = (
    (0, "SIMPLE_PINHOLE", 3),
    (1, "PINHOLE", 4),
    (2, "SIMPLE_RADIAL", 4),
    (3, "RADIAL", 5),
    (4, "OPENCV", 8),
    (5, "OPENCV_FISHEYE", 8),
    (6, "FULL_OPENCV", 12),
    (7, "FOV", 5),
    (8, "SIMPLE_RADIAL_FISHEYE", 4),
    (9, "RADIAL_FISHEYE", 5),
    (10, "THIN_PRISM_FISHEYE", 12),
)
CAMERA_MODEL_IDS = {model_id: model_name for model_id, model_name, _ in CAMERA_MODELS}
CAMERA_MODEL_NAME_TO_ID = {model_name: model_id for model_id, model_name, _ in CAMERA_MODELS}
CAMERA_MODEL_PARAM_COUNT = {model_id: param_count for model_id, _, param_count in CAMERA_MODELS}


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
            param_count = CAMERA_MODEL_PARAM_COUNT[model_id]
            params = np.frombuffer(fid.read(8 * param_count), dtype=np.float64)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODEL_IDS[model_id],
                width=width,
                height=height,
                params=params.copy(),
            )
    return cameras


def write_cameras_binary(cameras: Dict[int, Camera], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras.values():
            model_id = CAMERA_MODEL_NAME_TO_ID[camera.model]
            fid.write(struct.pack("<iiQQ", camera.id, model_id, camera.width, camera.height))
            fid.write(struct.pack(f"<{len(camera.params)}d", *camera.params))


def read_images_binary(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with path.open("rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]

        for _ in range(num_images):
            header = struct.unpack("<i7di", fid.read(4 + 8 * 7 + 4))
            image_id = header[0]
            qvec = np.array(header[1:5], dtype=np.float64)
            tvec = np.array(header[5:8], dtype=np.float64)
            camera_id = header[8]
            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            image_name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point_ids = np.empty(num_points2d, dtype=np.int64)
            for idx in range(num_points2d):
                x, y, point_id = struct.unpack("<ddq", fid.read(24))
                xys[idx] = (x, y)
                point_ids[idx] = point_id
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point_ids,
            )
    return images


def write_images_binary(images: Dict[int, Image], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images.values():
            fid.write(struct.pack("<i", image.id))
            fid.write(struct.pack("<4d", *image.qvec))
            fid.write(struct.pack("<3d", *image.tvec))
            fid.write(struct.pack("<i", image.camera_id))
            fid.write(image.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.point3D_ids)))
            for (x, y), point_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", float(x), float(y), int(point_id)))


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    with path.open("rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            packed = struct.unpack("<Q3d3Bd", fid.read(8 + 24 + 3 + 8))
            point_id = packed[0]
            xyz = np.array(packed[1:4], dtype=np.float64)
            rgb = np.array(packed[4:7], dtype=np.uint8)
            error = float(packed[7])
            track_len = struct.unpack("<Q", fid.read(8))[0]
            image_ids = np.empty(track_len, dtype=np.int32)
            point2d_idxs = np.empty(track_len, dtype=np.int32)
            for idx in range(track_len):
                image_id, point2d_idx = struct.unpack("<ii", fid.read(8))
                image_ids[idx] = image_id
                point2d_idxs[idx] = point2d_idx
            points[point_id] = Point3D(
                id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2d_idxs,
            )
    return points


def write_points3d_binary(points: Dict[int, Point3D], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(points)))
        for point in points.values():
            fid.write(struct.pack("<Q3d3Bd", point.id, *point.xyz, *point.rgb, float(point.error)))
            fid.write(struct.pack("<Q", len(point.image_ids)))
            for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
                fid.write(struct.pack("<ii", int(image_id), int(point2d_idx)))


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:

    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert a world-to-camera rotation matrix to a (w, x, y, z) quaternion."""

    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project vec onto the plane orthogonal to normal."""
    return vec - normal * np.dot(vec, normal)



def _collect_point_samples(points: Dict[int, Point3D], max_points: int = 200000) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    xyz_stack = np.stack([pt.xyz for pt in points.values()], axis=0)
    if xyz_stack.shape[0] <= max_points:
        return xyz_stack
    idx = np.linspace(0, xyz_stack.shape[0] - 1, max_points, dtype=np.int64)
    return xyz_stack[idx]


def compute_alignment(points: Dict[int, Point3D], images: Dict[int, Image]) -> np.ndarray:
    """Estimate a global rotation that aligns the scene with canonical XYZ axes."""

    if len(images) < 3:
        raise ValueError("Need at least three images to estimate alignment")

    # 从相机姿态提取方向与中心
    forward_vectors = []
    up_vectors = []
    centers = []
    for image in images.values():
        R_c2w = image.rotation_matrix().T
        forward_vectors.append(R_c2w @ np.array([0.0, 0.0, -1.0]))
        up_vectors.append(R_c2w @ np.array([0.0, 1.0, 0.0]))
        centers.append(-R_c2w @ image.tvec)

    centers = np.stack(centers, axis=0)
    centers_mean = centers.mean(axis=0)
    centers_centered = centers - centers_mean

    # 对相机中心做 PCA，最小方差方向往往对应竖直
    cov_centers = np.cov(centers_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_centers)
    order = np.argsort(eigvals)[::-1]
    center_primary_candidate = eigvecs[:, order[0]]
    center_secondary_candidate = eigvecs[:, order[1]]
    z_axis = eigvecs[:, order[2]].copy()

    # 点云 PCA 进一步提供车身长轴/法向信息
    xyz_stack = _collect_point_samples(points)
    normal_points = None
    point_primary_candidate = None
    point_secondary_candidate = None
    if xyz_stack.size:
        xyz_centered = xyz_stack - xyz_stack.mean(axis=0)
        cov_xyz = np.cov(xyz_centered.T)
        eigvals_p, eigvecs_p = np.linalg.eigh(cov_xyz)
        order_p = np.argsort(eigvals_p)
        normal_points = eigvecs_p[:, order_p[0]]
        point_primary_candidate = eigvecs_p[:, order_p[-1]]
        point_secondary_candidate = eigvecs_p[:, order_p[-2]]

    up_mean = np.mean(up_vectors, axis=0)
    forward_sum = np.sum([project_to_plane(vec, z_axis) for vec in forward_vectors], axis=0)
    if normal_points is not None and np.dot(normal_points, z_axis) < 0:
        z_axis = -z_axis
    if np.linalg.norm(up_mean) > 1e-6 and np.dot(up_mean, z_axis) < 0:
        z_axis = -z_axis
    if z_axis[2] < 0:
        z_axis = -z_axis
    z_axis = normalize(z_axis)

    # 优先使用点云水平 PCA 获取车身长轴
    x_axis = np.zeros(3, dtype=np.float64)
    y_hint = np.zeros(3, dtype=np.float64)

    if point_primary_candidate is not None:
        primary = project_to_plane(point_primary_candidate, z_axis)
        if np.linalg.norm(primary) > 1e-6:
            x_axis = normalize(primary)
            if point_secondary_candidate is not None:
                y_hint = project_to_plane(point_secondary_candidate, z_axis)

    if np.linalg.norm(x_axis) < 1e-6:
        primary_centers = project_to_plane(center_primary_candidate, z_axis)
        if np.linalg.norm(primary_centers) > 1e-6:
            x_axis = normalize(primary_centers)
            y_hint = project_to_plane(center_secondary_candidate, z_axis)

    if np.linalg.norm(x_axis) < 1e-6:
        fallback = project_to_plane(np.array([1.0, 0.0, 0.0]), z_axis)
        if np.linalg.norm(fallback) < 1e-6:
            fallback = project_to_plane(np.array([0.0, 1.0, 0.0]), z_axis)
        x_axis = normalize(fallback)

    # 使用相机前向与点云长轴决定正方向
    if np.linalg.norm(forward_sum) > 1e-6 and np.dot(forward_sum, x_axis) > 0:
        x_axis = -x_axis
    if point_primary_candidate is not None:
        primary_dir = project_to_plane(point_primary_candidate, z_axis)
        if np.linalg.norm(primary_dir) > 1e-6 and np.dot(primary_dir, x_axis) < 0:
            x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-6 and np.linalg.norm(y_hint) > 1e-6:
        y_axis = np.cross(z_axis, normalize(y_hint))
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)
    if np.dot(np.cross(z_axis, x_axis), y_axis) < 0:
        y_axis = -y_axis

    R_align = np.stack([normalize(x_axis), y_axis, z_axis], axis=0)
    if np.linalg.det(R_align) < 0:
        R_align[1] *= -1.0
    return R_align


def rotate_points(points: Dict[int, Point3D], R_align: np.ndarray) -> Dict[int, Point3D]:
    rotated: Dict[int, Point3D] = {}
    for point_id, point in points.items():
        rotated_xyz = R_align @ point.xyz
        rotated[point_id] = Point3D(
            id=point.id,
            xyz=rotated_xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        )
    return rotated


def rotate_images(images: Dict[int, Image], R_align: np.ndarray) -> Dict[int, Image]:
    rotated: Dict[int, Image] = {}
    for image_id, image in images.items():
        R_old = image.rotation_matrix()
        R_new = R_old @ R_align.T
        q_new = rotmat2qvec(R_new)

        # 先将平移转换为世界系相机中心，再应用全局旋转，最后回写新的平移
        C_old = -R_old.T @ image.tvec
        C_new = R_align @ C_old
        t_new = -R_new @ C_new

        rotated[image_id] = image.copy_with(qvec=q_new, tvec=t_new.astype(np.float64))
    return rotated


def save_alignment(R_align: np.ndarray, output_dir: Path) -> None:
    text = ["# Rows describe new axes expressed in the original coordinate system."]
    for row in R_align:
        text.append(" ".join(f"{value:.9f}" for value in row))
    (output_dir / "alignment_matrix.txt").write_text("\n".join(text) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate COLMAP sparse model to align axes")
    parser.add_argument("--input", required=True, type=Path, help="Path to sparse/0 directory")
    parser.add_argument("--output", required=True, type=Path, help="Destination directory")
    parser.add_argument("--copy-cameras", action="store_true", help="Copy cameras.bin instead of rewriting")
    return parser.parse_args()


def reorient_sparse_model(input_dir: Path, output_dir: Path, *, copy_cameras: bool = False) -> np.ndarray:
    """Rotate a COLMAP sparse model directory and write results to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras_binary(input_dir / "cameras.bin")
    images = read_images_binary(input_dir / "images.bin")
    points = read_points3d_binary(input_dir / "points3D.bin")

    R_align = compute_alignment(points, images)
    rotated_points = rotate_points(points, R_align)
    rotated_images = rotate_images(images, R_align)

    if copy_cameras:
        shutil.copy2(input_dir / "cameras.bin", output_dir / "cameras.bin")
    else:
        write_cameras_binary(cameras, output_dir / "cameras.bin")
    write_images_binary(rotated_images, output_dir / "images.bin")
    write_points3d_binary(rotated_points, output_dir / "points3D.bin")
    save_alignment(R_align, output_dir)
    return R_align


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output

    R_align = reorient_sparse_model(input_dir, output_dir, copy_cameras=args.copy_cameras)

    np.set_printoptions(precision=6, suppress=True)
    print("Alignment rotation (rows = new axes in old coordinates):")
    print(R_align)
    print(f"Aligned sparse model written to {output_dir}")


if __name__ == "__main__":
    main()
