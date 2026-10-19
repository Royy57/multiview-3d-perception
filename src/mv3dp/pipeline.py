"""End-to-end pipeline: SfM -> dense -> detection -> 3D fusion -> 6DoF pose -> viewer.

Each stage reads the previous stages' outputs from ``output_dir`` so stages can
be re-run individually (e.g. tweak fusion parameters without redoing SfM).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from mv3dp.config import PipelineConfig, save_config
from mv3dp.detection.detector import (
    YoloDetector, detect_images, draw_detections, load_detections, save_detections,
)
from mv3dp.evaluation.alignment import T_cam_world, align_to_reference, rescale_reference
from mv3dp.fusion.multiview import fuse_detections, load_instances, save_instances
from mv3dp.pose.pnp import (
    PoseEstimator, build_object_models, draw_pose, model_points_in_object_frame, pose_errors,
)
from mv3dp.reconstruction import dense as dense_mod
from mv3dp.reconstruction.sparse import (
    list_images, pinhole_cameras, run_sfm, sparse_points, split_holdout, undistort,
)
from mv3dp.viz.scene import save_boxes_ply, write_html_viewer

log = logging.getLogger(__name__)

STAGES = ("sparse", "dense", "detect", "fuse", "pose", "viz")


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.out = cfg.out
        self.out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out / "config.yaml")

    # ----------------------------------------------------------------- paths
    @property
    def sfm_dir(self) -> Path:
        return self.out / "sfm"

    @property
    def undistorted_dir(self) -> Path:
        return self.out / "undistorted"

    @property
    def dense_ply(self) -> Path:
        return self.out / "dense" / "fused.ply"

    @property
    def detections_json(self) -> Path:
        return self.out / "detections" / "detections.json"

    @property
    def instances_json(self) -> Path:
        return self.out / "objects" / "instances.json"

    def split(self) -> tuple[list[str], list[str]]:
        return split_holdout(list_images(self.cfg.image_dir), self.cfg.sparse.holdout_every)

    def sparse_model_dir(self) -> Path:
        return Path((self.sfm_dir / "best_model.txt").read_text().strip())

    def load_undistorted(self) -> pycolmap.Reconstruction:
        return pycolmap.Reconstruction(str(self.undistorted_dir / "sparse"))

    # ---------------------------------------------------------------- stages
    def run(self, stages: list[str] | None = None) -> None:
        for stage in stages or STAGES:
            if stage not in STAGES:
                raise ValueError(f"Unknown stage '{stage}', choose from {STAGES}")
            t0 = time.time()
            log.info("=== stage: %s ===", stage)
            getattr(self, f"stage_{stage}")()
            log.info("=== %s done in %.1fs ===", stage, time.time() - t0)

    def stage_sparse(self) -> None:
        train, test = self.split()
        log.info("%d images for SfM, %d held out for pose evaluation", len(train), len(test))
        model_dir = run_sfm(self.cfg.image_dir, self.sfm_dir, self.cfg.sparse, train)
        (self.sfm_dir / "best_model.txt").write_text(str(model_dir.resolve()))
        undistort(self.cfg.image_dir, model_dir, self.undistorted_dir)

    def stage_dense(self) -> None:
        if not self.cfg.dense.enabled:
            log.info("Dense reconstruction disabled")
            return
        rec = self.load_undistorted()
        out_dir = self.dense_ply.parent
        if self.cfg.dense.method == "colmap":
            result = dense_mod.run_colmap_mvs(self.undistorted_dir, out_dir, self.cfg.dense)
        elif self.cfg.dense.method == "sgbm":
            result = dense_mod.run_sgbm(rec, pinhole_cameras(rec), self.undistorted_dir / "images",
                                        out_dir, self.cfg.dense)
        else:
            raise ValueError(f"Unknown dense method '{self.cfg.dense.method}'")
        dense_mod.save_ply(self.dense_ply, result.points, result.colors)

    def stage_detect(self) -> None:
        rec = self.load_undistorted()
        names = sorted(img.name for img in rec.images.values() if img.has_pose)
        image_dir = self.undistorted_dir / "images"
        detections = detect_images(image_dir, names, self.cfg.detection)
        self.detections_json.parent.mkdir(parents=True, exist_ok=True)
        save_detections(self.detections_json, detections)
        preview = self.detections_json.parent / "preview"
        preview.mkdir(exist_ok=True)
        for name in names[:: max(1, len(names) // 8)]:
            img = cv2.imread(str(image_dir / name))
            cv2.imwrite(str(preview / name), draw_detections(img, detections[name]))

    def stage_fuse(self) -> None:
        rec = self.load_undistorted()
        cams = pinhole_cameras(rec)
        detections = load_detections(self.detections_json)
        dense_points, depth_dir = None, None
        if self.dense_ply.exists():
            dense_points, _ = dense_mod.load_ply(self.dense_ply)
            d = self.dense_ply.parent / "depth"
            depth_dir = d if d.exists() else None
        instances = fuse_detections(rec, cams, detections, self.cfg.fusion,
                                    self.cfg.detection.mask_erosion_px, dense_points, depth_dir)
        self.instances_json.parent.mkdir(parents=True, exist_ok=True)
        save_instances(self.instances_json, instances)
        save_boxes_ply(self.instances_json.parent / "boxes.ply", instances)
        summary = {}
        for inst in instances:
            summary[inst.class_name] = summary.get(inst.class_name, 0) + 1
        (self.instances_json.parent / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info("Objects: %s", summary)

    def stage_pose(self) -> None:
        if not self.cfg.pose.enabled:
            return
        _, test = self.split()
        if not test:
            log.info("No held-out images; skipping pose estimation")
            return
        distorted = pycolmap.Reconstruction(str(self.sparse_model_dir()))
        undistorted = self.load_undistorted()
        instances = load_instances(self.instances_json)
        models = build_object_models(distorted, self.sfm_dir / "database.db", instances)
        camera = next(iter(distorted.cameras.values()))
        estimator = PoseEstimator(camera, self.cfg.sparse, self.cfg.pose)
        detector = YoloDetector(self.cfg.detection)

        reference_poses, metrics = None, {}
        if self.cfg.reference_model_dir.exists():
            reference = pycolmap.Reconstruction(str(self.cfg.reference_model_dir))
            rescale_reference(reference, camera.width)
            _, xyz, _ = sparse_points(undistorted)
            scale = dense_mod.scene_extent(xyz)
            alignment = align_to_reference(undistorted, reference, scale)
            reference_poses = alignment.reference_poses
            metrics["reconstruction"] = alignment.metrics
            log.info("Reconstruction vs reference: %s", alignment.metrics)

        pose_dir = self.out / "pose"
        (pose_dir / "vis").mkdir(parents=True, exist_ok=True)
        results = []
        for name in test:
            img = cv2.imread(str(self.cfg.image_dir / name))
            dets = detector(img, name)
            kps, desc = estimator.features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            canvas = img.copy()
            for model in models:
                est = estimator.estimate(name, kps, desc, model, dets)
                if not est.detection_box:
                    continue  # object class not detected in this image
                size = model.instance.box.size
                if reference_poses is not None and name in reference_poses:
                    T_gt = T_cam_world(reference_poses[name]) @ model.T_world_obj
                    if est.success:
                        est.errors = pose_errors(est.T_cam_obj, T_gt, model_points_in_object_frame(model),
                                                 model.instance.box.diameter, self.cfg.pose.add_threshold)
                    if _visible(T_gt, camera):
                        draw_pose(canvas, T_gt, size, camera, (0, 0, 255), 1)
                if est.success:
                    draw_pose(canvas, est.T_cam_obj, size, camera, (0, 255, 0), 2)
                results.append(est)
            cv2.imwrite(str(pose_dir / "vis" / name), canvas)

        (pose_dir / "poses.json").write_text(json.dumps([r.to_dict() for r in results], indent=1))
        metrics["pose"] = summarize_pose(results)
        (self.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        log.info("Pose metrics: %s", metrics["pose"])

    def stage_viz(self) -> None:
        rec = self.load_undistorted()
        if self.dense_ply.exists():
            points, colors = dense_mod.load_ply(self.dense_ply)
        else:
            _, points, colors = sparse_points(rec)
        instances = load_instances(self.instances_json) if self.instances_json.exists() else []
        write_html_viewer(self.out / "viewer.html", points, colors, instances,
                          list(pinhole_cameras(rec).values()),
                          title=f"{Path(self.cfg.scene_dir).name}: {len(instances)} objects")
        log.info("Viewer written to %s", self.out / "viewer.html")


def _visible(T_cam_obj: np.ndarray, camera: pycolmap.Camera) -> bool:
    return T_cam_obj[2, 3] > 0


def summarize_pose(results) -> dict:
    attempted = [r for r in results if r.detection_box]
    solved = [r for r in attempted if r.success]
    evaluated = [r for r in solved if "add" in r.errors]
    out = {
        "queries_with_detection": len(attempted),
        "pnp_success": len(solved),
        "pnp_success_rate": len(solved) / max(len(attempted), 1),
    }
    if evaluated:
        rot = np.array([r.errors["rot_err_deg"] for r in evaluated])
        add_rel = np.array([r.errors["add_rel"] for r in evaluated])
        out.update({
            "evaluated": len(evaluated),
            "rot_err_median_deg": float(np.median(rot)),
            "add_rel_median": float(np.median(add_rel)),
            "add_0.1d_accuracy_of_solved": float(np.mean([r.errors["add_success"] for r in evaluated])),
            "add_0.1d_accuracy_of_queries": float(
                sum(r.errors["add_success"] for r in evaluated) / max(len(attempted), 1)
            ),
        })
    return out
