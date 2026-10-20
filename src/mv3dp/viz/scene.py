"""Exports for inspecting results: PLY files for MeshLab/CloudCompare and a
self-contained interactive HTML viewer (Plotly) that opens in any browser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from mv3dp.detection.detector import color_for_class
from mv3dp.fusion.multiview import ObjectInstance
from mv3dp.geometry import PinholeCamera


def boxes_lineset(instances: list[ObjectInstance]) -> o3d.geometry.LineSet:
    points, lines, colors = [], [], []
    for inst in instances:
        base = len(points)
        points.extend(inst.box.corners().tolist())
        rgb = color_for_class(inst.class_id)[::-1] / 255.0
        for a, b in inst.box.EDGES:
            lines.append([base + a, base + b])
            colors.append(rgb.tolist())
    ls = o3d.geometry.LineSet()
    if points:
        ls.points = o3d.utility.Vector3dVector(np.array(points))
        ls.lines = o3d.utility.Vector2iVector(np.array(lines))
        ls.colors = o3d.utility.Vector3dVector(np.array(colors))
    return ls


def save_boxes_ply(path: Path, instances: list[ObjectInstance]) -> None:
    o3d.io.write_line_set(str(path), boxes_lineset(instances))


def _hex(rgb: np.ndarray) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def write_html_viewer(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    instances: list[ObjectInstance],
    cameras: list[PinholeCamera] | None = None,
    max_points: int = 150_000,
    title: str = "Object-aware 3D reconstruction",
) -> None:
    import plotly.graph_objects as go

    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    traces = [
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=1.5, color=[_hex(c) for c in colors]),
            name="point cloud", hoverinfo="skip",
        )
    ]
    for inst in instances:
        corners = inst.box.corners()
        xs, ys, zs = [], [], []
        for a, b in inst.box.EDGES:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]
        color = _hex(color_for_class(inst.class_id)[::-1])
        label = f"{inst.class_name} #{inst.instance_id} ({inst.score:.2f}, {len(inst.views)} views)"
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=5),
                                   name=label, hoverinfo="name"))
        top = corners[4:].mean(axis=0)
        traces.append(go.Scatter3d(x=[top[0]], y=[top[1]], z=[top[2]], mode="text",
                                   text=[inst.class_name], textfont=dict(color=color, size=12),
                                   showlegend=False, hoverinfo="skip"))
    if cameras:
        traces.append(_camera_trace(cameras, points))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#111", font=dict(color="#eee"),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def _camera_trace(cameras: list[PinholeCamera], points: np.ndarray):
    import plotly.graph_objects as go

    scale = 0.03 * float(np.linalg.norm(np.ptp(np.percentile(points, [5, 95], axis=0), axis=0)))
    xs, ys, zs = [], [], []
    for cam in cameras:
        w, h = cam.width, cam.height
        corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        rays = np.c_[corners_px, np.ones(4)] @ np.linalg.inv(cam.K).T
        rays /= rays[:, 2:3]
        world = (rays * scale - cam.t) @ cam.R
        c = cam.center
        for k in range(4):
            xs += [c[0], world[k, 0], world[(k + 1) % 4, 0], None]
            ys += [c[1], world[k, 1], world[(k + 1) % 4, 1], None]
            zs += [c[2], world[k, 2], world[(k + 1) % 4, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#888", width=1),
                        name="cameras", visible="legendonly", hoverinfo="skip")


def draw_boxes_on_image(image: np.ndarray, cam: PinholeCamera, instances: list[ObjectInstance],
                        thickness: int = 2) -> np.ndarray:
    """Project the fused 3D boxes into an (undistorted) view and draw them."""
    import cv2

    canvas = image.copy()
    for inst in instances:
        corners = inst.box.corners()
        uv, z = cam.project(corners)
        if np.any(z <= 1e-6):
            continue
        if not np.any(cam.in_image(uv, z)):
            continue
        color = tuple(int(c) for c in color_for_class(inst.class_id))
        pts = np.round(uv).astype(int)
        for a, b in inst.box.EDGES:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
        top = pts[4:].min(axis=0)
        label = f"{inst.class_name} #{inst.instance_id}"
        cv2.putText(canvas, label, (int(pts[4:, 0].mean()) - 20, max(int(top[1]) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas
