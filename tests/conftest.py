import numpy as np
import pytest

from mv3dp.geometry import PinholeCamera


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """cam_from_world (R, t) for a camera at ``center`` looking at ``target``.

    The image +Y axis points along -up so the camera is upright.
    """
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


@pytest.fixture
def ring_cameras() -> list[PinholeCamera]:
    """12 upright cameras on a ring around the origin, all looking at it."""
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    cams = []
    for k in range(12):
        a = 2 * np.pi * k / 12
        c = np.array([4 * np.cos(a), 4 * np.sin(a), 1.5])
        R, t = look_at(c, np.zeros(3))
        cams.append(PinholeCamera(f"img{k:02d}.png", K, R, t, 640, 480))
    return cams
