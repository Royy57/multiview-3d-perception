import cv2
import numpy as np

from mv3dp.config import DenseConfig
from mv3dp.geometry import PinholeCamera
from mv3dp.reconstruction.dense import merge_depths, pair_depth, splat_depth


def _texture(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (h // 4, w // 4), dtype=np.uint8)
    tex = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)


def _plane_pair(depth: float, baseline: np.ndarray):
    """Two views of a fronto-parallel textured plane at ``depth`` (ref camera z)."""
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    ref = PinholeCamera("ref", K, np.eye(3), np.zeros(3), 640, 480)
    src = PinholeCamera("src", K, np.eye(3), -baseline, 640, 480)  # centre at +baseline
    tex = _texture(480, 640)
    # Homography induced by the plane z = depth: x_src ~ K (I + t n^T / d) K^-1 x_ref
    n = np.array([0, 0, 1.0])
    H = K @ (np.eye(3) + np.outer(src.t, n) / depth) @ np.linalg.inv(K)
    src_img = cv2.warpPerspective(tex, H, (640, 480))
    return ref, src, tex, src_img


def _plane_points(ref: PinholeCamera, depth: float) -> np.ndarray:
    uv = np.stack(np.meshgrid(np.linspace(100, 540, 10), np.linspace(100, 380, 10)), -1).reshape(-1, 2)
    return ref.backproject(uv, np.full(len(uv), depth))


def test_sgbm_depth_on_plane_right_neighbour():
    ref, src, ref_img, src_img = _plane_pair(4.0, np.array([0.3, 0.0, 0.0]))
    depth = pair_depth(ref_img, src_img, ref, src, _plane_points(ref, 4.0), DenseConfig())
    assert depth is not None
    valid = depth > 0
    assert valid.mean() > 0.3
    assert abs(np.median(depth[valid]) - 4.0) < 0.05


def test_sgbm_depth_on_plane_left_neighbour():
    # Source to the left of the reference exercises the flipped-matching path.
    ref, src, ref_img, src_img = _plane_pair(4.0, np.array([-0.3, 0.0, 0.0]))
    depth = pair_depth(ref_img, src_img, ref, src, _plane_points(ref, 4.0), DenseConfig())
    assert depth is not None
    valid = depth > 0
    assert valid.mean() > 0.3
    assert abs(np.median(depth[valid]) - 4.0) < 0.05


def test_splat_depth_keeps_nearest(ring_cameras):
    cam = ring_cameras[0]
    X_cam = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    depth = splat_depth(X_cam, cam)
    assert depth[240, 320] == 2.0


def test_merge_depths_requires_agreement():
    a = np.array([[1.0, 1.0]], np.float32)
    b = np.array([[1.005, 2.0]], np.float32)
    merged = merge_depths([a, b], 0.02)
    assert merged[0, 0] > 0 and merged[0, 1] == 0


def test_sgbm_depth_on_plane_vertical_baselines():
    for baseline in ([0.0, 0.3, 0.0], [0.0, -0.3, 0.0]):
        ref, src, ref_img, src_img = _plane_pair(4.0, np.array(baseline))
        depth = pair_depth(ref_img, src_img, ref, src, _plane_points(ref, 4.0), DenseConfig())
        assert depth is not None
        valid = depth > 0
        assert valid.mean() > 0.3
        assert abs(np.median(depth[valid]) - 4.0) < 0.05
