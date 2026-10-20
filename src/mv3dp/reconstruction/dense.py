"""Dense reconstruction from calibrated views.

Two back-ends:

* ``sgbm``: for every reference view, pick well-conditioned neighbours from the
  SfM co-visibility graph, rectify each pair from the known poses, run OpenCV
  semi-global block matching, and convert disparity to a depth map. Depth maps
  are then fused into one point cloud, keeping only points that are
  geometrically consistent across several views. Runs on CPU on any OS.
* ``colmap``: COLMAP PatchMatch stereo + fusion through the ``colmap``
  executable (requires a CUDA-enabled COLMAP build, e.g. the Windows release).
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from mv3dp.config import DenseConfig
from mv3dp.geometry import PinholeCamera

log = logging.getLogger(__name__)


@dataclass
class DenseResult:
    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) uint8
    depth_dir: Path | None = None


# --------------------------------------------------------------------------- #
# Neighbour selection
# --------------------------------------------------------------------------- #
def select_neighbors(
    rec: pycolmap.Reconstruction,
    cams: dict[int, PinholeCamera],
    min_angle_deg: float,
    max_angle_deg: float,
    num_neighbors: int,
    max_points_per_pair: int = 200,
    seed: int = 0,
) -> dict[int, list[int]]:
    """Rank source views for each reference view.

    Score = number of shared 3D points, for pairs whose median triangulation
    angle lies inside [min_angle, max_angle] (too small is noisy, too large
    makes rectification and matching fail).
    """
    rng = np.random.default_rng(seed)
    shared: dict[tuple[int, int], list[int]] = defaultdict(list)
    xyz = {}
    for pid, point in rec.points3D.items():
        ids = sorted({el.image_id for el in point.track.elements if el.image_id in cams})
        if len(ids) < 2:
            continue
        xyz[pid] = point.xyz
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                shared[(ids[a], ids[b])].append(pid)

    centers = {i: c.center for i, c in cams.items()}
    scored: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (i, j), pids in shared.items():
        if len(pids) < 10:
            continue
        sample = pids if len(pids) <= max_points_per_pair else list(rng.choice(pids, max_points_per_pair, replace=False))
        P = np.array([xyz[p] for p in sample])
        ri = P - centers[i]
        rj = P - centers[j]
        cos = np.sum(ri * rj, axis=1) / (np.linalg.norm(ri, axis=1) * np.linalg.norm(rj, axis=1))
        angle = np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))
        if not (min_angle_deg <= angle <= max_angle_deg):
            continue
        score = float(len(pids))
        scored[i].append((score, j))
        scored[j].append((score, i))

    return {i: [j for _, j in sorted(scored.get(i, []), reverse=True)[:num_neighbors]] for i in cams}


# --------------------------------------------------------------------------- #
# Two-view stereo
# --------------------------------------------------------------------------- #
@dataclass
class RectifiedPair:
    R1: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    map_ref: tuple[np.ndarray, np.ndarray]
    map_src: tuple[np.ndarray, np.ndarray]
    vertical: bool  # baseline along image y: images are transposed before matching
    flipped: bool  # source lies left of (or above) the reference: images are mirrored

    @property
    def focal(self) -> float:
        return float(self.P1[0, 0])

    @property
    def baseline(self) -> float:
        axis = 1 if self.vertical else 0
        return float(abs(self.P2[axis, 3] / self.P2[axis, axis]))


def rectify_pair(ref: PinholeCamera, src: PinholeCamera) -> RectifiedPair | None:
    """Rectify (ref, src) from their known poses. Returns None for ill-conditioned pairs."""
    size = (ref.width, ref.height)
    R = src.R @ ref.R.T
    T = src.t - R @ ref.t
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        ref.K, None, src.K, None, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])  # OpenCV chose vertical rectification
    if P1[0, 0] > 2.5 * ref.K[0, 0] or P1[0, 0] < 0.4 * ref.K[0, 0]:
        return None  # rotation too strong: rectified images would be badly warped
    map_ref = cv2.initUndistortRectifyMap(ref.K, None, R1, P1, size, cv2.CV_32FC1)
    map_src = cv2.initUndistortRectifyMap(src.K, None, R2, P2, size, cv2.CV_32FC1)
    axis = 1 if vertical else 0
    return RectifiedPair(R1, P1, P2, map_ref, map_src, vertical, flipped=bool(P2[axis, 3] > 0))


def expected_disparity_range(pair: RectifiedPair, ref: PinholeCamera, points: np.ndarray) -> tuple[int, int] | None:
    """Disparity search range from sparse points seen by both views."""
    Xc = ref.world_to_cam(points) @ pair.R1.T
    z = Xc[:, 2]
    z = z[z > 1e-6]
    if len(z) < 5:
        return None
    disp = pair.focal * pair.baseline / z
    lo, hi = np.percentile(disp, [1, 99])
    lo = max(0, int(np.floor(lo * 0.75)))
    num = int(np.ceil((hi * 1.25 - lo) / 16.0)) * 16
    return lo, int(np.clip(num, 16, 512))


def pair_depth(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    ref: PinholeCamera,
    src: PinholeCamera,
    shared_points: np.ndarray,
    cfg: DenseConfig,
) -> np.ndarray | None:
    """Depth map (camera z) for the reference view, in its own pixel grid."""
    pair = rectify_pair(ref, src)
    if pair is None:
        return None
    rng = expected_disparity_range(pair, ref, shared_points)
    if rng is None:
        return None
    min_disp, num_disp = rng
    if min_disp + num_disp > 0.6 * (ref.height if pair.vertical else ref.width):
        return None  # search range larger than the image: unreliable pair

    left = cv2.remap(ref_img, *pair.map_ref, cv2.INTER_LINEAR)
    right = cv2.remap(src_img, *pair.map_src, cv2.INTER_LINEAR)
    # SGBM searches along +x in the left image: transpose vertical pairs and
    # mirror pairs whose source lies on the "wrong" side, then undo afterwards.
    if pair.vertical:
        left, right = cv2.transpose(left), cv2.transpose(right)
    if pair.flipped:
        left, right = cv2.flip(left, 1), cv2.flip(right, 1)

    block = cfg.sgbm_block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=cfg.sgbm_uniqueness_ratio,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    if pair.flipped:
        disp = cv2.flip(disp, 1)
    if pair.vertical:
        disp = cv2.transpose(disp)
    valid = disp > max(min_disp, 0.5)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return None
    d = disp[v, u]
    f, cx, cy = pair.P1[0, 0], pair.P1[0, 2], pair.P1[1, 2]
    z = f * pair.baseline / d
    X_rect = np.stack([(u - cx) * z / f, (v - cy) * z / f, z], axis=1)
    X_cam = X_rect @ pair.R1  # R1 maps cam -> rect, so R1^T maps back
    return splat_depth(X_cam, ref)


def splat_depth(X_cam: np.ndarray, cam: PinholeCamera) -> np.ndarray:
    """Z-buffer camera-frame points into a depth image (0 = empty)."""
    z = X_cam[:, 2]
    keep = z > 1e-6
    X_cam, z = X_cam[keep], z[keep]
    uv = (X_cam[:, :2] / z[:, None]) @ cam.K[:2, :2].T + cam.K[:2, 2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.full((cam.height, cam.width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), z.astype(np.float32))
    depth[np.isinf(depth)] = 0
    return depth


def merge_depths(depths: list[np.ndarray], rel_tol: float) -> np.ndarray:
    """Per-pixel median of agreeing depth hypotheses from different source views."""
    if len(depths) == 1:
        return depths[0]
    stack = np.stack(depths).astype(np.float32)
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
        agree = np.abs(stack - med) <= rel_tol * med
        count = np.sum(agree, axis=0)
        merged = np.where(count >= 2, med, 0)
    return np.nan_to_num(merged).astype(np.float32)


# --------------------------------------------------------------------------- #
# Depth map fusion
# --------------------------------------------------------------------------- #
def fuse_depth_maps(
    depths: dict[int, np.ndarray],
    images: dict[int, np.ndarray],
    cams: dict[int, PinholeCamera],
    neighbors: dict[int, list[int]],
    cfg: DenseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    all_pts, all_rgb = [], []
    for i, depth in tqdm(depths.items(), desc="fusing depth maps"):
        v, u = np.nonzero(depth > 0)
        if len(u) == 0:
            continue
        z = depth[v, u]
        P = cams[i].backproject(np.stack([u, v], axis=1).astype(float), z.astype(float))
        consistent = np.ones(len(P), dtype=np.int32)
        for k in neighbors.get(i, []):
            if k not in depths:
                continue
            uv, zk = cams[k].project(P)
            inside = cams[k].in_image(uv, zk)
            uk = np.clip(np.round(uv[:, 0]).astype(int), 0, cams[k].width - 1)
            vk = np.clip(np.round(uv[:, 1]).astype(int), 0, cams[k].height - 1)
            dk = depths[k][vk, uk]
            ok = inside & (dk > 0) & (np.abs(dk - zk) <= cfg.consistency_rel_depth * zk)
            consistent += ok
        keep = consistent >= cfg.min_consistent_views
        all_pts.append(P[keep])
        all_rgb.append(images[i][v[keep], u[keep]][:, ::-1])  # BGR -> RGB
    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts), np.concatenate(all_rgb)


def clean_point_cloud(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).round().astype(np.uint8)


def scene_extent(points: np.ndarray) -> float:
    lo, hi = np.percentile(points, [5, 95], axis=0)
    return float(np.linalg.norm(hi - lo))


def resize_view(cam: PinholeCamera, max_size: int) -> tuple[PinholeCamera, float]:
    scale = min(1.0, max_size / max(cam.width, cam.height))
    if scale >= 1.0:
        return cam, 1.0
    K = cam.K.copy()
    K[:2] *= scale
    w, h = int(round(cam.width * scale)), int(round(cam.height * scale))
    return PinholeCamera(cam.name, K, cam.R, cam.t, w, h), scale


def run_sgbm(
    rec: pycolmap.Reconstruction,
    cams: dict[int, PinholeCamera],
    image_dir: Path,
    out_dir: Path,
    cfg: DenseConfig,
) -> DenseResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    neighbors = select_neighbors(
        rec, cams, cfg.min_triangulation_angle_deg, cfg.max_triangulation_angle_deg, cfg.num_neighbors
    )
    small = {i: resize_view(c, cfg.max_image_size)[0] for i, c in cams.items()}
    images = {}
    for i, cam in small.items():
        img = cv2.imread(str(image_dir / cam.name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_dir / cam.name)
        images[i] = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)

    observed: dict[int, set[int]] = defaultdict(set)
    for pid, point in rec.points3D.items():
        for el in point.track.elements:
            observed[el.image_id].add(pid)

    depths: dict[int, np.ndarray] = {}
    for i in tqdm(small, desc="SGBM depth maps"):
        hyps = []
        for j in neighbors.get(i, []):
            if len(hyps) == 2:
                break
            common = list(observed[i] & observed[j])
            if len(common) < 10:
                continue
            pts = np.array([rec.points3D[p].xyz for p in common])
            d = pair_depth(images[i], images[j], small[i], small[j], pts, cfg)
            if d is not None:
                hyps.append(d)
        if hyps:
            depths[i] = merge_depths(hyps, cfg.consistency_rel_depth * 2)
            np.save(depth_dir / f"{Path(small[i].name).stem}.npy", depths[i].astype(np.float16))

    log.info("Computed %d/%d depth maps", len(depths), len(small))
    points, colors = fuse_depth_maps(depths, images, small, neighbors, cfg)
    if len(points) == 0:
        raise RuntimeError("Dense fusion produced no points")
    voxel = cfg.voxel_size_rel * scene_extent(points)
    points, colors = clean_point_cloud(points, colors, voxel)
    if len(points) > cfg.max_points:
        idx = np.random.default_rng(0).choice(len(points), cfg.max_points, replace=False)
        points, colors = points[idx], colors[idx]
    log.info("Dense cloud: %d points", len(points))
    return DenseResult(points, colors, depth_dir)


def run_colmap_mvs(undistorted_dir: Path, out_dir: Path, cfg: DenseConfig) -> DenseResult:
    """COLMAP PatchMatch stereo + fusion (CUDA build of COLMAP required)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fused = out_dir / "fused_colmap.ply"
    exe = cfg.colmap_executable
    subprocess.run(
        [exe, "patch_match_stereo", "--workspace_path", str(undistorted_dir),
         "--PatchMatchStereo.max_image_size", str(cfg.max_image_size),
         "--PatchMatchStereo.geom_consistency", "true"],
        check=True,
    )
    subprocess.run(
        [exe, "stereo_fusion", "--workspace_path", str(undistorted_dir), "--output_path", str(fused)],
        check=True,
    )
    pcd = o3d.io.read_point_cloud(str(fused))
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).round().astype(np.uint8)
    return DenseResult(points, colors, None)


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).round().astype(np.uint8)
