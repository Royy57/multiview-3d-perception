"""Camera and rigid-body geometry helpers.

Conventions follow COLMAP: a world point X maps to camera coordinates as
``x_cam = R @ X + t`` (``cam_from_world``), and the camera looks down +Z with
+X right and +Y down in the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    """A calibrated, undistorted pinhole view with its pose."""

    name: str
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) cam_from_world rotation
    t: np.ndarray  # (3,)   cam_from_world translation
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    @property
    def T_cam_world(self) -> np.ndarray:
        return to_homogeneous(self.R, self.t)

    def world_to_cam(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points. Returns pixel coords (N, 2) and depths (N,)."""
        Xc = self.world_to_cam(X)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (Xc[:, :2] / z[:, None]) @ self.K[:2, :2].T + self.K[:2, 2]
        return uv, z

    def in_image(self, uv: np.ndarray, z: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (
            (z > 1e-6)
            & (uv[:, 0] >= margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 0] < self.width - margin)
            & (uv[:, 1] < self.height - margin)
        )

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift pixels with depth (camera z) to world points."""
        Kinv = np.linalg.inv(self.K)
        rays = np.c_[uv, np.ones(len(uv))] @ Kinv.T
        Xc = rays * depth[:, None]
        return (Xc - self.t) @ self.R


def to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return to_homogeneous(R.T, -R.T @ t)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def estimate_up_vector(cameras: list[PinholeCamera]) -> np.ndarray:
    """Estimate the world 'up' direction from roughly upright cameras.

    Handheld captures keep the camera's image-down axis (+Y) close to gravity,
    so the average of the cameras' -Y axes in world coordinates points up.
    """
    downs = np.stack([cam.R.T @ np.array([0.0, 1.0, 0.0]) for cam in cameras])
    up = -downs.mean(axis=0)
    return up / np.linalg.norm(up)


def frame_from_up(up: np.ndarray) -> np.ndarray:
    """Rotation whose third column is ``up`` (a gravity-aligned world frame)."""
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up], axis=1)
