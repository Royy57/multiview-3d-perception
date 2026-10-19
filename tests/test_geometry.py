import numpy as np
from scipy.spatial.transform import Rotation

from mv3dp.evaluation.alignment import umeyama
from mv3dp.fusion.boxes import fit_gravity_box
from mv3dp.geometry import estimate_up_vector, frame_from_up, invert, rotation_angle_deg, to_homogeneous


def test_project_backproject_roundtrip(ring_cameras):
    cam = ring_cameras[0]
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (100, 3))
    uv, z = cam.project(X)
    assert np.all(cam.in_image(uv, z))
    np.testing.assert_allclose(cam.backproject(uv, z), X, atol=1e-9)


def test_invert_and_rotation_angle():
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    T = to_homogeneous(R, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(invert(T) @ T, np.eye(4), atol=1e-12)
    R2 = Rotation.from_euler("z", 15, degrees=True).as_matrix() @ R
    assert abs(rotation_angle_deg(R, R2) - 15) < 1e-6


def test_up_vector_from_upright_cameras(ring_cameras):
    up = estimate_up_vector(ring_cameras)
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-9)
    G = frame_from_up(up)
    np.testing.assert_allclose(G.T @ G, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(G[:, 2], up)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    R = Rotation.from_euler("xyz", [5, -40, 70], degrees=True).as_matrix()
    dst = 2.5 * src @ R.T + np.array([0.3, -1.0, 4.0])
    sim = umeyama(src, dst)
    assert abs(sim.s - 2.5) < 1e-9
    np.testing.assert_allclose(sim.apply(src), dst, atol=1e-9)


def test_gravity_box_fits_rotated_box():
    rng = np.random.default_rng(2)
    size = np.array([2.0, 1.0, 0.5])
    local = rng.uniform(-0.5, 0.5, (5000, 3)) * size
    yaw = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    center = np.array([1.0, -2.0, 0.7])
    points = local @ yaw.T + center
    box = fit_gravity_box(points, np.eye(3), percentile=0.5)
    np.testing.assert_allclose(np.sort(box.size), np.sort(size), rtol=0.05)
    np.testing.assert_allclose(box.center, center, atol=0.03)
    assert box.contains(points, margin=0.02).mean() > 0.97
    np.testing.assert_allclose(box.rotation[:, 2], [0, 0, 1], atol=1e-9)


def test_box_iou_3d():
    from mv3dp.fusion.boxes import Box3D

    a = Box3D(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.eye(3))
    b = Box3D(np.array([1.0, 0, 0]), np.array([2.0, 2.0, 2.0]), np.eye(3))
    assert abs(a.iou(a) - 1.0) < 1e-9
    assert abs(a.iou(b) - 1 / 3) < 0.02
    small = Box3D(np.zeros(3), np.array([0.2, 0.2, 0.2]), np.eye(3))
    assert a.iou(small) < 0.01
