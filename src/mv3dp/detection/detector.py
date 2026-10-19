"""2D instance detection with Ultralytics YOLOv11 (segmentation variant)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mv3dp.config import DetectionConfig

log = logging.getLogger(__name__)


@dataclass
class Detection:
    image: str
    class_id: int
    class_name: str
    score: float
    box: list[float]  # x1, y1, x2, y2 in pixels
    polygons: list[list[list[float]]] = field(default_factory=list)  # instance outline(s)

    def mask(self, height: int, width: int, erosion_px: int = 0, scale: float = 1.0) -> np.ndarray:
        """Binary mask at (height, width). Falls back to the box when no outline exists."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.polygons:
            polys = [np.round(np.asarray(p) * scale).astype(np.int32) for p in self.polygons if len(p) >= 3]
            cv2.fillPoly(mask, polys, 1)
        else:
            x1, y1, x2, y2 = (np.asarray(self.box) * scale).round().astype(int)
            mask[max(y1, 0) : y2, max(x1, 0) : x2] = 1
        if erosion_px > 0:
            k = 2 * erosion_px + 1
            mask = cv2.erode(mask, np.ones((k, k), np.uint8))
        return mask.astype(bool)


class YoloDetector:
    def __init__(self, cfg: DetectionConfig):
        from ultralytics import YOLO  # imported lazily: torch start-up is slow

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        names = self.model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        self.keep_ids = None
        if cfg.classes:
            lookup = {v: k for k, v in self.names.items()}
            missing = [c for c in cfg.classes if c not in lookup]
            if missing:
                raise ValueError(f"Classes not known by {cfg.model}: {missing}")
            self.keep_ids = [lookup[c] for c in cfg.classes]

    def __call__(self, image: np.ndarray | str | Path, name: str) -> list[Detection]:
        result = self.model.predict(
            source=image if isinstance(image, np.ndarray) else str(image),
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.image_size,
            device=self.cfg.device or None,
            classes=self.keep_ids,
            retina_masks=True,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        outlines = result.masks.xy if result.masks is not None else [None] * len(boxes)
        for box, score, cls, outline in zip(boxes, scores, classes, outlines):
            polygons = [outline.tolist()] if outline is not None and len(outline) >= 3 else []
            detections.append(
                Detection(name, int(cls), self.names[int(cls)], float(score), box.tolist(), polygons)
            )
        return detections


def detect_images(image_dir: Path, names: list[str], cfg: DetectionConfig) -> dict[str, list[Detection]]:
    detector = YoloDetector(cfg)
    out: dict[str, list[Detection]] = {}
    for name in tqdm(names, desc="YOLO detection"):
        out[name] = detector(image_dir / name, name)
    total = sum(len(v) for v in out.values())
    log.info("Detected %d instances in %d images", total, len(names))
    return out


def save_detections(path: Path, detections: dict[str, list[Detection]]) -> None:
    payload = {name: [asdict(d) for d in dets] for name, dets in detections.items()}
    path.write_text(json.dumps(payload))


def load_detections(path: Path) -> dict[str, list[Detection]]:
    payload = json.loads(Path(path).read_text())
    return {name: [Detection(**d) for d in dets] for name, dets in payload.items()}


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Overlay masks, boxes and labels for quick visual checks."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        color = tuple(int(c) for c in color_for_class(det.class_id))
        mask = det.mask(h, w)
        canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{det.class_name} {det.score:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def color_for_class(class_id: int) -> np.ndarray:
    """Stable, distinct BGR color per class id."""
    hue = (class_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
