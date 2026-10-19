"""Fuse per-view 2D detections into 3D object instances.

The SfM tracks give exact 2D-3D links: every 3D point knows which keypoint in
which image observed it. That lets us

1. label each sparse 3D point with a class when most of the images observing
   it put it inside a mask of that class,
2. link detections across images that share labelled 3D points, and take the
   connected components as object hypotheses,
3. split hypotheses that are spatially separate (e.g. two chairs that were
   linked by a few boundary points), and
4. optionally add dense points that project inside the instance's masks, then
   fit a gravity-aligned 3D box.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from mv3dp.config import FusionConfig
from mv3dp.detection.detector import Detection
from mv3dp.fusion.boxes import Box3D, fit_gravity_box
from mv3dp.geometry import PinholeCamera, estimate_up_vector, frame_from_up

log = logging.getLogger(__name__)


@dataclass
class ObjectInstance:
    instance_id: int
    class_id: int
    class_name: str
    score: float
    box: Box3D
    views: list[str]  # image names the object was detected in
    detections: list[tuple[str, int]]  # (image name, index into that image's detections)
    point_ids: list[int] = field(default_factory=list)  # sparse SfM point ids
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # points used for the box

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "box": self.box.to_dict(),
            "num_views": len(self.views),
            "views": self.views,
            "detections": [list(d) for d in self.detections],
            "num_sparse_points": len(self.point_ids),
            "num_points": int(len(self.points)),
            "point_ids": [int(p) for p in self.point_ids],
        }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())


def radius_clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Connected components of the radius graph (single-linkage clustering)."""
    if len(points) == 0:
        return []
    uf = UnionFind(len(points))
    for a, b in cKDTree(points).query_pairs(radius):
        uf.union(a, b)
    return sorted((np.array(g) for g in uf.groups()), key=len, reverse=True)


def median_spacing(points: np.ndarray, max_samples: int = 20000, seed: int = 0) -> float:
    if len(points) < 2:
        return 1.0
    sample = points
    if len(points) > max_samples:
        sample = points[np.random.default_rng(seed).choice(len(points), max_samples, replace=False)]
    d, _ = cKDTree(points).query(sample, k=2)
    return float(np.median(d[:, 1]))


@dataclass
class _Det:
    gid: int
    image_id: int
    index: int
    det: Detection
    pids: set[int]


def _collect(
    rec: pycolmap.Reconstruction,
    cams: dict[int, PinholeCamera],
    detections: dict[str, list[Detection]],
    erosion_px: int,
) -> list[_Det]:
    name_to_id = {c.name: i for i, c in cams.items()}
    out: list[_Det] = []
    for name, dets in detections.items():
        image_id = name_to_id.get(name)
        if image_id is None or not dets:
            continue
        cam = cams[image_id]
        image = rec.images[image_id]
        xy = np.array([p.xy for p in image.points2D]).reshape(-1, 2)
        p3d = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D], dtype=np.int64)
        has = p3d >= 0
        xy, p3d = xy[has], p3d[has]
        u = np.clip(np.round(xy[:, 0]).astype(int), 0, cam.width - 1)
        v = np.clip(np.round(xy[:, 1]).astype(int), 0, cam.height - 1)
        for k, det in enumerate(dets):
            mask = det.mask(cam.height, cam.width, erosion_px)
            inside = mask[v, u]
            out.append(_Det(len(out), image_id, k, det, set(p3d[inside].tolist())))
    return out


def fuse_detections(
    rec: pycolmap.Reconstruction,
    cams: dict[int, PinholeCamera],
    detections: dict[str, list[Detection]],
    cfg: FusionConfig,
    erosion_px: int = 3,
    dense_points: np.ndarray | None = None,
    depth_dir: Path | None = None,
) -> list[ObjectInstance]:
    dets = _collect(rec, cams, detections, erosion_px)
    if not dets:
        return []
    detected_images = {d.image_id for d in dets} | {
        i for i, c in cams.items() if c.name in detections
    }

    # 1. Point labelling by multi-view voting.
    votes: dict[int, Counter] = defaultdict(Counter)
    for d in dets:
        for pid in d.pids:
            votes[pid][d.det.class_id] += 1
    point_label: dict[int, int] = {}
    for pid, counter in votes.items():
        track_images = {el.image_id for el in rec.points3D[pid].track.elements} & detected_images
        cls, n = counter.most_common(1)[0]
        if n >= cfg.min_point_votes and n / max(len(track_images), 1) >= cfg.point_vote_ratio:
            point_label[pid] = cls

    labelled = [{p for p in d.pids if point_label.get(p) == d.det.class_id} for d in dets]

    # 2. Link detections of the same class that share labelled points.
    dets_of_point: dict[int, list[int]] = defaultdict(list)
    for d, pts in zip(dets, labelled):
        for pid in pts:
            dets_of_point[pid].append(d.gid)
    shared: Counter = Counter()
    for gids in dets_of_point.values():
        for a in range(len(gids)):
            for b in range(a + 1, len(gids)):
                ga, gb = gids[a], gids[b]
                if dets[ga].image_id != dets[gb].image_id:
                    shared[(min(ga, gb), max(ga, gb))] += 1
    uf = UnionFind(len(dets))
    for (ga, gb), n in shared.items():
        overlap = n / max(min(len(labelled[ga]), len(labelled[gb])), 1)
        if n >= cfg.min_shared_points and overlap >= cfg.min_overlap_ratio:
            uf.union(ga, gb)

    # 3. Spatially split each component into instances.
    all_xyz = np.array([p.xyz for p in rec.points3D.values()])
    radius = cfg.cluster_radius_factor * median_spacing(all_xyz)
    G = frame_from_up(estimate_up_vector(list(cams.values())))

    candidates: list[ObjectInstance] = []
    for group in uf.groups():
        cls = dets[group[0]].det.class_id
        count: Counter = Counter()
        for g in group:
            count.update(labelled[g])
        pids = np.array([p for p, n in count.items() if n >= cfg.min_point_votes], dtype=np.int64)
        if len(pids) < cfg.min_points:
            continue
        xyz = np.array([rec.points3D[int(p)].xyz for p in pids])
        for cluster in radius_clusters(xyz, radius):
            if len(cluster) < cfg.min_points:
                break
            members = set(pids[cluster].tolist())
            member_dets = [g for g in group if len(labelled[g] & members) >= max(2, 0.3 * len(labelled[g]))]
            views = sorted({cams[dets[g].image_id].name for g in member_dets})
            if len(views) < cfg.min_views:
                continue
            pts = xyz[cluster]
            candidates.append(
                ObjectInstance(
                    instance_id=-1,
                    class_id=cls,
                    class_name=dets[group[0]].det.class_name,
                    score=float(np.mean([dets[g].det.score for g in member_dets])),
                    box=fit_gravity_box(pts, G, cfg.box_percentile),
                    views=views,
                    detections=[(cams[dets[g].image_id].name, dets[g].index) for g in member_dets],
                    point_ids=sorted(members),
                    points=pts,
                )
            )

    # Merging grows boxes, which can make further fragments mergeable: repeat.
    instances, previous = candidates, -1
    while len(instances) != previous:
        previous = len(instances)
        instances = merge_duplicates(instances, G, cfg.box_percentile, cfg.merge_overlap)
    instances = [o for o in instances if detection_ratio(o, cams) >= cfg.min_detection_ratio]

    # 4. Densify with dense points that agree with the masks.
    if dense_points is not None and len(dense_points) and cfg.use_dense_points:
        name_to_id = {c.name: i for i, c in cams.items()}
        for inst in instances:
            extra = dense_points_for_instance(inst, dense_points, cams, name_to_id, detections,
                                              erosion_px, cfg.point_vote_ratio, depth_dir)
            if len(extra) >= cfg.min_points:
                inst.points = np.concatenate([inst.points, extra])
                inst.box = fit_gravity_box(inst.points, G, cfg.box_percentile)

    instances = suppress_cross_class(instances, detections, cfg.cross_class_iou_2d, cfg.cross_class_iou_3d)
    instances.sort(key=lambda o: (o.class_name, -o.score))
    for k, inst in enumerate(instances):
        inst.instance_id = k
    log.info("Fused %d object instances from %d detections", len(instances), len(dets))
    return instances


def merge_duplicates(cands: list[ObjectInstance], G: np.ndarray, percentile: float,
                     min_overlap: float = 0.3) -> list[ObjectInstance]:
    """Merge same-class fragments of one object.

    Two same-class instances are fragments when one box contains much of the
    other's points, when they came from mostly the same 2D detections (a single
    mask whose 3D points were split by a gap in the sparse cloud), or when they
    are close together and never detected as two separate objects in the same
    image (a large object seen from different sides, like a plant).
    """
    cands = sorted(cands, key=lambda o: -len(o.points))
    merged: list[ObjectInstance] = []
    for c in cands:
        target = None
        for m in merged:
            if m.class_id != c.class_id:
                continue
            a = m.box.contains(c.points).mean()
            b = c.box.contains(m.points).mean()
            shared = len(set(m.detections) & set(c.detections)) / max(min(len(m.detections), len(c.detections)), 1)
            if max(a, b) > min_overlap or shared >= 0.5 or _same_object_seen_apart(m, c):
                target = m
                break
        if target is None:
            merged.append(c)
            continue
        target.points = np.concatenate([target.points, c.points])
        target.point_ids = sorted(set(target.point_ids) | set(c.point_ids))
        target.views = sorted(set(target.views) | set(c.views))
        target.detections = sorted(set(target.detections) | set(c.detections))
        target.score = max(target.score, c.score)
        target.box = fit_gravity_box(target.points, G, percentile)
    return merged


def _same_object_seen_apart(a: ObjectInstance, b: ObjectInstance) -> bool:
    det_a = dict(a.detections)
    for name, idx in b.detections:
        if name in det_a and det_a[name] != idx:
            return False  # two distinct detections in one image: two objects
    gap = np.linalg.norm(a.box.center - b.box.center)
    return gap < 0.5 * (a.box.diameter + b.box.diameter)


def detection_ratio(inst: ObjectInstance, cams: dict[int, PinholeCamera]) -> float:
    """Views with a detection / views in which the whole box projects into the image.

    Occlusion is ignored, so this underestimates how often a real object was
    detected; it is only meant to reject sporadic false positives.
    """
    corners = inst.box.corners()
    visible = 0
    for cam in cams.values():
        uv, z = cam.project(corners)
        if np.all(z > 0) and np.all(cam.in_image(uv, z, margin=-0.05 * cam.width)):
            visible += 1
    return len(inst.views) / max(visible, len(inst.views), 1)


def box_iou_2d(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def suppress_cross_class(instances: list[ObjectInstance], detections: dict[str, list[Detection]],
                         min_iou_2d: float, min_iou_3d: float, min_shared_views: int = 3) -> list[ObjectInstance]:
    """Resolve one object detected under two labels (e.g. a jar as 'cup' and 'bottle').

    In images where both instances were detected, their 2D boxes must overlap
    strongly (IoU) for them to be the same object. When the two labels were
    never given in the same images (the detector flip-flops between them), the
    3D boxes must overlap instead. An object resting inside another (an apple in
    a bowl) has a small IoU either way and is kept. The label seen in more views
    wins.
    """
    def boxes(inst: ObjectInstance) -> dict[str, list[float]]:
        return {name: detections[name][idx].box for name, idx in inst.detections}

    ranked = sorted(instances, key=lambda o: (len(o.views), o.score), reverse=True)
    kept: list[ObjectInstance] = []
    for inst in ranked:
        mine = boxes(inst)
        duplicate = False
        for k in kept:
            if k.class_id == inst.class_id:
                continue
            theirs = boxes(k)
            common = set(mine) & set(theirs)
            if len(common) >= min_shared_views:
                same = np.median([box_iou_2d(mine[n], theirs[n]) for n in common]) >= min_iou_2d
            else:
                same = inst.box.iou(k.box) >= min_iou_3d
            if same:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def dense_points_for_instance(
    inst: ObjectInstance,
    dense: np.ndarray,
    cams: dict[int, PinholeCamera],
    name_to_id: dict[str, int],
    detections: dict[str, list[Detection]],
    erosion_px: int,
    vote_ratio: float,
    depth_dir: Path | None,
    margin: float = 0.15,
    max_views: int = 15,
) -> np.ndarray:
    """Dense points near the instance that fall inside its masks in most views."""
    cand = dense[inst.box.contains(dense, margin)]
    if len(cand) == 0:
        return cand
    hits = np.zeros(len(cand), dtype=np.int32)
    seen = np.zeros(len(cand), dtype=np.int32)
    by_image: dict[str, list[int]] = defaultdict(list)
    for name, idx in inst.detections:
        by_image[name].append(idx)
    for name in list(by_image)[:max_views]:
        cam = cams[name_to_id[name]]
        mask = np.zeros((cam.height, cam.width), dtype=bool)
        for idx in by_image[name]:
            mask |= detections[name][idx].mask(cam.height, cam.width, erosion_px)
        uv, z = cam.project(cand)
        vis = cam.in_image(uv, z)
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, cam.width - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, cam.height - 1)
        if depth_dir is not None:
            vis &= ~_occluded(depth_dir, cam, u, v, z)
        seen += vis
        hits += vis & mask[v, u]
    keep = (seen >= 2) & (hits >= vote_ratio * np.maximum(seen, 1))
    return cand[keep]


def _occluded(depth_dir: Path, cam: PinholeCamera, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    path = depth_dir / f"{Path(cam.name).stem}.npy"
    if not path.exists():
        return np.zeros(len(u), dtype=bool)
    depth = np.load(path).astype(np.float32)
    if depth.shape != (cam.height, cam.width):
        depth = cv2.resize(depth, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
    d = depth[v, u]
    return (d > 0) & (d < z * 0.97)


def save_instances(path: Path, instances: list[ObjectInstance]) -> None:
    path.write_text(json.dumps([inst.to_dict() for inst in instances], indent=1))
    np.savez_compressed(path.with_suffix(".npz"), **{str(i.instance_id): i.points for i in instances})


def load_instances(path: Path) -> list[ObjectInstance]:
    data = json.loads(Path(path).read_text())
    pts_path = Path(path).with_suffix(".npz")
    pts = np.load(pts_path) if pts_path.exists() else {}
    out = []
    for d in data:
        out.append(
            ObjectInstance(
                instance_id=d["instance_id"],
                class_id=d["class_id"],
                class_name=d["class_name"],
                score=d["score"],
                box=Box3D.from_dict(d["box"]),
                views=d["views"],
                detections=[tuple(x) for x in d["detections"]],
                point_ids=d.get("point_ids", []),
                points=pts[str(d["instance_id"])] if str(d["instance_id"]) in pts else np.zeros((0, 3)),
            )
        )
    return out
