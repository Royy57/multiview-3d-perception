"""6DoF object pose from a single new image via 2D-3D matching and PnP.

Each fused object instance becomes a feature model: its sparse 3D points carry
the SIFT descriptors of every keypoint that observed them during SfM. For a new
image, YOLO localises the object, SIFT features inside the box are matched to
the model, and PnP + RANSAC returns the pose of the object in the camera frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from mv3dp.config import PoseConfig, SparseConfig
from mv3dp.detection.detector import Detection
from mv3dp.fusion.multiview import ObjectInstance
from mv3dp.geometry import invert, to_homogeneous
from mv3dp.reconstruction.sparse import sift_options

log = logging.getLogger(__name__)


def _normalize(desc: np.ndarray) -> np.ndarray:
    desc = desc.astype(np.float32)
    return desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)


@dataclass
class ObjectModel:
    instance: ObjectInstance
    xyz: np.ndarray  # (M, 3) world coordinates, one row per descriptor
    descriptors: np.ndarray  # (M, 128) unit-norm float32
    point_ids: np.ndarray  # (M,) sparse point id per descriptor

    @property
    def T_world_obj(self) -> np.ndarray:
        return to_homogeneous(self.instance.box.rotation, self.instance.box.center)

    @property
    def model_points(self) -> np.ndarray:
        """Unique 3D points, used for the ADD metric."""
        _, idx = np.unique(self.point_ids, return_index=True)
        return self.xyz[idx]


def build_object_models(
    rec: pycolmap.Reconstruction,
    database_path: Path,
    instances: list[ObjectInstance],
    max_desc_per_point: int = 8,
) -> list[ObjectModel]:
    """Attach SfM descriptors to each instance's sparse 3D points."""
    db = pycolmap.Database(str(database_path))
    cache: dict[int, np.ndarray] = {}
    models = []
    for inst in instances:
        xyz, desc, pids = [], [], []
        for pid in inst.point_ids:
            if pid not in rec.points3D:
                continue
            point = rec.points3D[pid]
            for el in list(point.track.elements)[:max_desc_per_point]:
                if el.image_id not in cache:
                    cache[el.image_id] = db.read_descriptors(el.image_id)
                d = cache[el.image_id]
                if el.point2D_idx < len(d):
                    xyz.append(point.xyz)
                    desc.append(d[el.point2D_idx])
                    pids.append(pid)
        if len(desc) >= 4:
            models.append(ObjectModel(inst, np.array(xyz), _normalize(np.array(desc)), np.array(pids)))
    db.close()
    return models


@dataclass
class PoseEstimate:
    image: str
    instance_id: int
    class_name: str
    success: bool
    num_matches: int = 0
    num_inliers: int = 0
    T_cam_obj: np.ndarray | None = None
    detection_box: list[float] | None = None
    errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "success": self.success,
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
            "T_cam_obj": None if self.T_cam_obj is None else self.T_cam_obj.tolist(),
            "detection_box": self.detection_box,
            **self.errors,
        }


class PoseEstimator:
    """Estimate object poses in images taken with the SfM camera."""

    def __init__(self, camera: pycolmap.Camera, sparse_cfg: SparseConfig, cfg: PoseConfig):
        self.camera = camera  # the distorted SfM camera, so raw images can be used
        self.cfg = cfg
        self.sift = pycolmap.Sift(sift_options(sparse_cfg), pycolmap.Device.cpu)
        self.focal = float(camera.mean_focal_length())

    def features(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kps, desc = self.sift.extract(np.ascontiguousarray(gray.astype(np.float32) / 255.0))
        return kps[:, :2], _normalize(desc)

    def estimate(
        self,
        image_name: str,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        model: ObjectModel,
        detections: list[Detection],
    ) -> PoseEstimate:
        """Try every detection of the model's class; keep the one with most PnP inliers."""
        inst = model.instance
        best = PoseEstimate(image_name, inst.instance_id, inst.class_name, success=False)
        for det in detections:
            if det.class_id != inst.class_id:
                continue
            x1, y1, x2, y2 = det.box
            pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            inside = (
                (keypoints[:, 0] >= x1 - pad_x) & (keypoints[:, 0] <= x2 + pad_x)
                & (keypoints[:, 1] >= y1 - pad_y) & (keypoints[:, 1] <= y2 + pad_y)
            )
            if inside.sum() < self.cfg.min_inliers:
                continue
            est = self._solve(keypoints[inside], descriptors[inside], model)
            est.image, est.detection_box = image_name, list(det.box)
            if est.num_inliers > best.num_inliers:
                best = est
        return best

    def _solve(self, kps: np.ndarray, desc: np.ndarray, model: ObjectModel) -> PoseEstimate:
        inst = model.instance
        out = PoseEstimate("", inst.instance_id, inst.class_name, success=False)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(desc, model.descriptors, k=2)
        q_idx, m_idx = [], []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < self.cfg.ratio_test * pair[1].distance:
                q_idx.append(pair[0].queryIdx)
                m_idx.append(pair[0].trainIdx)
        out.num_matches = len(q_idx)
        if len(q_idx) < self.cfg.min_inliers:
            return out

        # Undistort with the SfM camera model: PnP then runs on normalised rays.
        rays = self.camera.cam_from_img(kps[q_idx].astype(np.float64))
        obj = model.xyz[m_idx].astype(np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, rays, np.eye(3), None,
            reprojectionError=self.cfg.ransac_reproj_error_px / self.focal,
            iterationsCount=2000, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok or inliers is None or len(inliers) < self.cfg.min_inliers:
            out.num_inliers = 0 if inliers is None else len(inliers)
            return out
        inl = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], rays[inl], np.eye(3), None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_cam_world = to_homogeneous(R, tvec[:, 0])
        out.success = True
        out.num_inliers = int(len(inl))
        out.T_cam_obj = T_cam_world @ model.T_world_obj
        return out


def pose_errors(T_est: np.ndarray, T_gt: np.ndarray, model_points_obj: np.ndarray, diameter: float,
                add_threshold: float) -> dict:
    """Rotation / translation errors and the ADD metric for one estimate."""
    from mv3dp.geometry import rotation_angle_deg

    P = np.c_[model_points_obj, np.ones(len(model_points_obj))]
    est = P @ T_est.T
    gt = P @ T_gt.T
    add = float(np.mean(np.linalg.norm(est[:, :3] - gt[:, :3], axis=1)))
    return {
        "rot_err_deg": rotation_angle_deg(T_est[:3, :3], T_gt[:3, :3]),
        "trans_err": float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])),
        "add": add,
        "add_rel": add / diameter,
        "add_success": bool(add < add_threshold * diameter),
    }


def model_points_in_object_frame(model: ObjectModel) -> np.ndarray:
    T_obj_world = invert(model.T_world_obj)
    P = model.model_points
    return P @ T_obj_world[:3, :3].T + T_obj_world[:3, 3]


def draw_pose(image: np.ndarray, T_cam_obj: np.ndarray, size: np.ndarray, camera: pycolmap.Camera,
              color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """Draw the object's 3D box and axes, projected with the (distorted) SfM camera."""
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    corners = signs * size / 2
    axes = np.array([[0, 0, 0], [size[0] / 2, 0, 0], [0, size[1] / 2, 0], [0, 0, size[2] / 2]])
    pts = np.r_[corners, axes]
    cam_pts = pts @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]
    if np.any(cam_pts[:, 2] <= 1e-6):
        return image
    uv = camera.img_from_cam(cam_pts[:, :2] / cam_pts[:, 2:3])
    uv = np.round(uv).astype(int)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for a, b in edges:
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), color, thickness, cv2.LINE_AA)
    for k, axis_color in zip((9, 10, 11), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(image, tuple(uv[8]), tuple(uv[k]), axis_color, thickness, cv2.LINE_AA)
    return image
