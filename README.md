# multiview-3d-perception

Multi-view 3D reconstruction combined with deep learning-based object detection.
A set of images of a scene is reconstructed with Structure-from-Motion and
Multi-View Stereo (COLMAP), objects are detected in every view with YOLOv11, and the
detections are fused across views into labelled 3D bounding boxes in the
reconstructed point cloud. For known objects, the 6DoF pose is recovered with PnP.

Runs natively on Windows (and Linux) with an NVIDIA GPU. No ROS required.

## Pipeline

1. **Sparse reconstruction**: feature extraction, matching, incremental SfM (pycolmap)
2. **Dense reconstruction**: MVS depth maps and fused point cloud
3. **2D detection**: YOLOv11 on every registered image
4. **Multi-view fusion**: detections lifted to 3D and clustered across views into 3D boxes
5. **6DoF pose**: 2D–3D correspondences and PnP + RANSAC for known objects
6. **Evaluation**: reconstruction accuracy (ETH3D) and pose accuracy (LINEMOD)

## Layout

```
configs/   pipeline and dataset configs
data/      downloaded datasets (git-ignored)
docs/      notes and figures
scripts/   entry points (download, run, evaluate)
src/mv3dp/ library code
tests/     unit tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

## License

MIT, see [LICENSE](LICENSE).
