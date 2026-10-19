"""Pipeline configuration loaded from YAML into typed dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class SparseConfig:
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 1600
    max_num_features: int = 8192
    # Exhaustive matching up to this many images, sequential matching beyond.
    exhaustive_max_images: int = 300
    sequential_overlap: int = 20
    use_gpu: bool = True
    # Every N-th image (sorted by name) is held out of SfM for pose evaluation,
    # following the usual Mip-NeRF 360 train/test split. 0 disables hold-out.
    holdout_every: int = 8


@dataclass
class DenseConfig:
    enabled: bool = True
    # "sgbm": OpenCV semi-global matching on rectified neighbour pairs (CPU, any OS).
    # "colmap": COLMAP PatchMatch MVS via the colmap executable (needs a CUDA build).
    method: str = "sgbm"
    colmap_executable: str = "colmap"
    max_image_size: int = 1000
    min_triangulation_angle_deg: float = 4.0
    max_triangulation_angle_deg: float = 25.0
    num_neighbors: int = 4
    sgbm_block_size: int = 5
    sgbm_uniqueness_ratio: int = 10
    # Multi-view geometric consistency used when fusing depth maps.
    consistency_rel_depth: float = 0.01
    min_consistent_views: int = 2
    voxel_size_rel: float = 0.002  # fraction of scene extent
    max_points: int = 3_000_000


@dataclass
class DetectionConfig:
    model: str = "yolo11m-seg.pt"
    confidence: float = 0.35
    iou: float = 0.5
    image_size: int = 1024
    device: str = ""  # "" lets Ultralytics choose (CUDA if available)
    # COCO class names to keep; empty keeps everything.
    classes: list[str] = field(default_factory=list)
    # Pixels to erode masks by before collecting points, to avoid edge bleeding.
    mask_erosion_px: int = 3


@dataclass
class FusionConfig:
    min_views: int = 3
    min_points: int = 15
    # Detections in two images are linked when they share enough 3D points.
    min_shared_points: int = 3
    min_overlap_ratio: float = 0.2
    # A 3D point is labelled with a class when this fraction of the images
    # observing it put it inside a mask of that class.
    point_vote_ratio: float = 0.5
    min_point_votes: int = 2
    # Spatial clustering radius, as a multiple of the median point spacing.
    cluster_radius_factor: float = 4.0
    box_percentile: float = 2.0
    use_dense_points: bool = True
    # Same-class instances are merged when this fraction of one's points lies
    # inside the other's box (fragments of one object, e.g. a plant's leaves).
    merge_overlap: float = 0.3
    # Instances of different classes that overlap this much are one object with
    # conflicting labels (keep the label seen in more views): median 2D box IoU
    # over views where both were detected, or 3D box IoU when never detected together.
    cross_class_iou_2d: float = 0.5
    cross_class_iou_3d: float = 0.2
    # Drop instances detected in only a small fraction of the views that see
    # them; these are usually sporadic false positives.
    min_detection_ratio: float = 0.05


@dataclass
class PoseConfig:
    enabled: bool = True
    ratio_test: float = 0.8
    ransac_reproj_error_px: float = 4.0
    min_inliers: int = 12
    # ADD success threshold as a fraction of the object diameter.
    add_threshold: float = 0.1


@dataclass
class PipelineConfig:
    scene_dir: str = "data/mipnerf360/counter"
    output_dir: str = "outputs/counter"
    seed: int = 0
    sparse: SparseConfig = field(default_factory=SparseConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)

    @property
    def image_dir(self) -> Path:
        return Path(self.scene_dir) / "images"

    @property
    def reference_model_dir(self) -> Path:
        return Path(self.scene_dir) / "reference_sparse"

    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        ftype = hints[name]
        kwargs[name] = _build(ftype, value) if is_dataclass(ftype) and isinstance(value, dict) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(PipelineConfig, data)


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
