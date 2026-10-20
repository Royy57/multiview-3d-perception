import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from mv3dp.config import FusionConfig, PoseConfig, SparseConfig
from mv3dp.detection.detector import Detection
from mv3dp.fusion.boxes import Box3D
from mv3dp.fusion.multiview import ObjectInstance, fuse_detections, radius_clusters
from mv3dp.geometry import to_homogeneous
from mv3dp.pose.pnp import ObjectModel, PoseEstimator, pose_errors, model_points_in_object_frame
from mv3dp.reconstruction.sparse import pinhole_cameras


def _dense_synthetic():
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_cameras = 1
    opts.num_images = 10
    opts.num_points3D = 2500
    opts.camera_model_id = pycolmap.CameraModelId.SIMPLE_PINHOLE
    opts.camera_params = [800.0, 512.0, 384.0]
    opts.num_points2D_without_point3D = 0
    pycolmap.set_random_seed(0)
    return pycolmap.synthesize_dataset(opts)


def test_radius_clusters_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (100, 3))
    b = rng.normal(0, 0.05, (80, 3)) + [2, 0, 0]
    clusters = radius_clusters(np.r_[a, b], 0.2)
    assert [len(c) for c in clusters[:2]] == [100, 80]


def test_fusion_recovers_two_objects_from_masks():
    rec = _dense_synthetic()
    cams = pinhole_cameras(rec)
    pids = np.array(list(rec.points3D))
    xyz = np.array([rec.points3D[p].xyz for p in pids])
    # Synthetic points lie on the unit sphere: the "objects" are two caps of it.
    centers = {"left": np.array([-0.6, 0.0, 0.8]), "right": np.array([0.6, 0.0, 0.8])}
    truth = {k: set(pids[np.linalg.norm(xyz - c, axis=1) < 0.35].tolist()) for k, c in centers.items()}

    detections = {}
    for image_id, cam in cams.items():
        image = rec.images[image_id]
        dets = []
        for k, members in truth.items():
            xy = np.array([p.xy for p in image.points2D if p.has_point3D() and p.point3D_id in members])
            hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
            x1, y1 = xy.min(0)
            x2, y2 = xy.max(0)
            dets.append(Detection(cam.name, 56, "chair", 0.9, [x1, y1, x2, y2], [hull.tolist()]))
        detections[cam.name] = dets

    instances = fuse_detections(rec, cams, detections, FusionConfig(min_points=10), erosion_px=0)
    assert len(instances) == 2
    for inst in instances:
        found = set(inst.point_ids)
        best = max(truth.values(), key=lambda t: len(t & found))
        assert len(best & found) / len(best) > 0.8  # recall
        assert len(best & found) / len(found) > 0.9  # precision
        assert len(inst.views) == len(cams)


def test_pnp_recovers_object_pose():
    rng = np.random.default_rng(3)
    camera = pycolmap.Camera.create(1, pycolmap.CameraModelId.SIMPLE_RADIAL, 700.0, 800, 600)
    camera.params = [700.0, 400.0, 300.0, 0.02]

    box = Box3D(np.array([0.2, -0.1, 0.5]), np.array([0.4, 0.3, 0.2]),
                Rotation.from_euler("z", 25, degrees=True).as_matrix())
    local = rng.uniform(-0.5, 0.5, (300, 3)) * box.size
    world = box.center + local @ box.rotation.T
    desc = rng.normal(size=(300, 128)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    inst = ObjectInstance(0, 41, "cup", 0.9, box, ["a", "b", "c"], [])
    model = ObjectModel(inst, world, desc, np.arange(300))

    R = Rotation.from_euler("xyz", [170, 10, -5], degrees=True).as_matrix()
    T_cam_world = to_homogeneous(R, np.array([0.0, 0.0, 2.0]) - R @ box.center)
    cam_pts = world @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    kps = camera.img_from_cam(cam_pts[:, :2] / cam_pts[:, 2:3]) + rng.normal(0, 0.3, (300, 2))
    q_desc = desc + rng.normal(0, 0.02, desc.shape).astype(np.float32)
    # 30 % outliers: random descriptors at random image positions.
    out_kps = rng.uniform([0, 0], [800, 600], (130, 2))
    out_desc = rng.normal(size=(130, 128)).astype(np.float32)
    kps = np.r_[kps, out_kps]
    q_desc = np.r_[q_desc, out_desc]
    q_desc /= np.linalg.norm(q_desc, axis=1, keepdims=True)

    estimator = PoseEstimator(camera, SparseConfig(), PoseConfig())
    det = Detection("q", 41, "cup", 0.9, [0, 0, 800, 600])
    est = estimator.estimate("q", kps, q_desc, model, [det])
    assert est.success and est.num_inliers > 200

    T_gt = T_cam_world @ model.T_world_obj
    err = pose_errors(est.T_cam_obj, T_gt, model_points_in_object_frame(model), box.diameter, 0.1)
    assert err["rot_err_deg"] < 0.5
    assert err["add_success"] and err["add_rel"] < 0.01
