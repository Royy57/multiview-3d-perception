"""Sparse reconstruction: SIFT features, matching, incremental SfM, undistortion."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from mv3dp.config import SparseConfig
from mv3dp.geometry import PinholeCamera

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: str | Path) -> list[str]:
    return sorted(p.name for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def split_holdout(names: list[str], every: int) -> tuple[list[str], list[str]]:
    """Mip-NeRF 360 convention: every ``every``-th image (by sorted name) is a test view."""
    if every <= 0:
        return names, []
    test = [n for i, n in enumerate(names) if i % every == 0]
    train = [n for i, n in enumerate(names) if i % every != 0]
    return train, test


def sift_options(cfg: SparseConfig) -> pycolmap.SiftExtractionOptions:
    opts = pycolmap.SiftExtractionOptions()
    opts.max_image_size = cfg.max_image_size
    opts.max_num_features = cfg.max_num_features
    return opts


def device(cfg: SparseConfig) -> pycolmap.Device:
    return pycolmap.Device.auto if cfg.use_gpu else pycolmap.Device.cpu


def run_sfm(image_dir: Path, work_dir: Path, cfg: SparseConfig, image_names: list[str]) -> Path:
    """Run feature extraction, matching and incremental mapping.

    Returns the directory of the largest reconstructed model.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    if database.exists():
        database.unlink()
    sparse_root = work_dir / "sparse"
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    sparse_root.mkdir()

    log.info("Extracting SIFT features from %d images", len(image_names))
    pycolmap.extract_features(
        str(database),
        str(image_dir),
        image_list=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=cfg.camera_model,
        sift_options=sift_options(cfg),
        device=device(cfg),
    )

    if len(image_names) <= cfg.exhaustive_max_images:
        log.info("Exhaustive matching")
        pycolmap.match_exhaustive(str(database), device=device(cfg))
    else:
        log.info("Sequential matching (overlap %d)", cfg.sequential_overlap)
        opts = pycolmap.SequentialMatchingOptions()
        opts.overlap = cfg.sequential_overlap
        pycolmap.match_sequential(str(database), matching_options=opts, device=device(cfg))

    log.info("Incremental mapping")
    models = pycolmap.incremental_mapping(str(database), str(image_dir), str(sparse_root))
    if not models:
        raise RuntimeError("SfM failed: no model was reconstructed")
    best_idx = max(models, key=lambda k: models[k].num_reg_images())
    best = models[best_idx]
    log.info("Best model %d: %d/%d images registered, %d points",
             best_idx, best.num_reg_images(), len(image_names), len(best.points3D))
    return sparse_root / str(best_idx)


def undistort(image_dir: Path, model_dir: Path, out_dir: Path, max_image_size: int = -1) -> Path:
    """Undistort images and model to a PINHOLE camera. Returns ``out_dir``."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    opts = pycolmap.UndistortCameraOptions()
    if max_image_size > 0:
        opts.max_image_size = max_image_size
    pycolmap.undistort_images(str(out_dir), str(model_dir), str(image_dir), undistort_options=opts)
    return out_dir


def pinhole_cameras(rec: pycolmap.Reconstruction) -> dict[int, PinholeCamera]:
    """Registered images of an undistorted reconstruction as pinhole cameras."""
    cams: dict[int, PinholeCamera] = {}
    for image_id, image in rec.images.items():
        if not image.has_pose:
            continue
        camera = rec.cameras[image.camera_id]
        if camera.model.name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"Expected an undistorted model, got {camera.model.name}")
        pose = image.cam_from_world
        cams[image_id] = PinholeCamera(
            name=image.name,
            K=camera.calibration_matrix(),
            R=pose.rotation.matrix(),
            t=np.asarray(pose.translation, dtype=float),
            width=camera.width,
            height=camera.height,
        )
    return cams


def sparse_points(rec: pycolmap.Reconstruction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, xyz, rgb) arrays for all 3D points."""
    ids = np.fromiter(rec.points3D.keys(), dtype=np.int64, count=len(rec.points3D))
    xyz = np.array([rec.points3D[i].xyz for i in ids], dtype=float).reshape(-1, 3)
    rgb = np.array([rec.points3D[i].color for i in ids], dtype=np.uint8).reshape(-1, 3)
    return ids, xyz, rgb
