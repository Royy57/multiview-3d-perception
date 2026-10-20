"""Gravity-aligned oriented 3D bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box3D:
    center: np.ndarray  # (3,) world
    size: np.ndarray  # (3,) extent along the box axes (length, width, height)
    rotation: np.ndarray  # (3, 3) columns are the box axes in world; third = up

    def corners(self) -> np.ndarray:
        """The 8 corners in world coordinates (bottom face first)."""
        signs = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
        )
        return self.center + (signs * self.size / 2) @ self.rotation.T

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) @ self.rotation

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Points inside the box grown by ``margin`` (a fraction of its size)."""
        local = np.abs(self.to_local(points))
        return np.all(local <= self.size * (0.5 + margin), axis=1)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.size))

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "size": self.size.tolist(), "rotation": self.rotation.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Box3D":
        return cls(np.asarray(d["center"]), np.asarray(d["size"]), np.asarray(d["rotation"]))

    def iou(self, other: "Box3D", samples_per_axis: int = 12) -> float:
        """Volumetric IoU, estimated on a regular grid of samples inside each box."""
        g = (np.arange(samples_per_axis) + 0.5) / samples_per_axis - 0.5
        unit = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        in_a = self.center + (unit * self.size) @ self.rotation.T
        in_b = other.center + (unit * other.size) @ other.rotation.T
        inter_a = other.contains(in_a).mean() * self.volume
        inter_b = self.contains(in_b).mean() * other.volume
        inter = 0.5 * (inter_a + inter_b)
        return float(inter / (self.volume + other.volume - inter))

    EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def fit_gravity_box(points: np.ndarray, world_from_gravity: np.ndarray, percentile: float = 2.0) -> Box3D:
    """Fit a box whose vertical axis follows gravity and whose yaw minimises area.

    ``world_from_gravity`` is a rotation with the up direction as third column.
    Extremes are taken at robust percentiles so a few stray points do not
    inflate the box.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to fit a box")
    G = world_from_gravity
    local = points @ G  # coordinates in the gravity frame
    xy = local[:, :2]

    # Trim horizontal outliers by distance from the median before fitting yaw.
    med = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - med, axis=1)
    core = xy[dist <= np.percentile(dist, 100 - percentile)] if len(xy) > 20 else xy

    (cx, cy), (w, h), angle = cv2.minAreaRect(core.astype(np.float32))
    theta = np.radians(angle)
    yaw = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Re-measure extents in the yawed frame with percentile trimming.
    rel = (xy - [cx, cy]) @ yaw
    lo_xy = np.percentile(rel, percentile, axis=0)
    hi_xy = np.percentile(rel, 100 - percentile, axis=0)
    lo_z, hi_z = np.percentile(local[:, 2], [percentile, 100 - percentile])

    mid_xy = np.array([cx, cy]) + yaw @ ((lo_xy + hi_xy) / 2)
    center_local = np.array([mid_xy[0], mid_xy[1], (lo_z + hi_z) / 2])
    size = np.array([hi_xy[0] - lo_xy[0], hi_xy[1] - lo_xy[1], hi_z - lo_z])
    size = np.maximum(size, 1e-6)

    R_local = np.eye(3)
    R_local[:2, :2] = yaw
    return Box3D(center=G @ center_local, size=size, rotation=G @ R_local)
