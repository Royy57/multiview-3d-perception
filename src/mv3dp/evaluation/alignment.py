"""Align our reconstruction to a reference model and evaluate camera poses.

SfM recovers geometry only up to a similarity transform, so before comparing
with the reference (e.g. the poses shipped with Mip-NeRF 360) we estimate the
7-DoF similarity between the two sets of camera centres (Umeyama).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap

from mv3dp.geometry import rotation_angle_deg, to_homogeneous


@dataclass
class Sim3:
    """x_ours = s * R @ x_ref + t"""

    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.s * X @ self.R.T + self.t


def umeyama(src: np.ndarray, dst: np.ndarray) -> Sim3:
    """Least-squares similarity with dst ≈ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def poses_by_name(rec: pycolmap.Reconstruction) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for image in rec.images.values():
        if image.has_pose:
            pose = image.cam_from_world
            out[image.name] = (pose.rotation.matrix(), np.asarray(pose.translation, dtype=float))
    return out


def center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


@dataclass
class AlignmentResult:
    sim3: Sim3
    reference_poses: dict[str, tuple[np.ndarray, np.ndarray]]  # in OUR frame and scale
    metrics: dict


def align_to_reference(ours: pycolmap.Reconstruction, reference: pycolmap.Reconstruction,
                       scene_scale: float) -> AlignmentResult:
    """Align by camera centres; report pose accuracy of our registered images.

    ``scene_scale`` (e.g. the extent of our point cloud) normalises the
    translation errors so they are comparable across scenes.
    """
    mine = poses_by_name(ours)
    ref = poses_by_name(reference)
    common = sorted(set(mine) & set(ref))
    if len(common) < 3:
        raise ValueError("Need at least 3 common images to align")
    c_ref = np.array([center(*ref[n]) for n in common])
    c_mine = np.array([center(*mine[n]) for n in common])
    sim = umeyama(c_ref, c_mine)

    # Express every reference pose in our frame: X_ref = R^T (X - t) / s.
    ref_in_ours = {}
    for name, (Rr, tr) in ref.items():
        R_new = Rr @ sim.R.T
        t_new = sim.s * tr - R_new @ sim.t
        ref_in_ours[name] = (R_new, t_new)

    pos_err = np.linalg.norm(sim.apply(c_ref) - c_mine, axis=1)
    rot_err = np.array([rotation_angle_deg(mine[n][0], ref_in_ours[n][0]) for n in common])
    metrics = {
        "num_registered": len(mine),
        "num_reference": len(ref),
        "num_common": len(common),
        "ate_rmse": float(np.sqrt(np.mean(pos_err**2))),
        "ate_rmse_rel": float(np.sqrt(np.mean(pos_err**2)) / scene_scale),
        "rot_err_median_deg": float(np.median(rot_err)),
        "rot_err_max_deg": float(np.max(rot_err)),
        "sim3_scale": sim.s,
    }
    return AlignmentResult(sim, ref_in_ours, metrics)


def T_cam_world(pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return to_homogeneous(*pose)


def rescale_reference(reference: pycolmap.Reconstruction, image_width: int) -> None:
    """Rescale reference cameras to the resolution of the images we use."""
    for camera in reference.cameras.values():
        if camera.width != image_width:
            camera.rescale(image_width / camera.width)
