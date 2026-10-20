"""Command-line entry points (also installed as ``mv3dp-run`` and ``mv3dp-download``)."""

import argparse
import logging

import yaml

from mv3dp.config import load_config
from mv3dp.data.mipnerf360 import INDOOR_SCENES, RESOLUTIONS, SCENES, download_scene
from mv3dp.pipeline import STAGES, Pipeline

RUN_HELP = """Run the object-aware reconstruction pipeline.

Examples:
    mv3dp-run --config configs/mipnerf360_counter.yaml
    mv3dp-run --config configs/mipnerf360_counter.yaml --stages fuse pose viz
    mv3dp-run --config configs/mipnerf360_counter.yaml --set detection.confidence=0.5
"""

DOWNLOAD_HELP = """Download Mip-NeRF 360 scenes (only the requested scene and resolution).

Example:
    mv3dp-download --scene counter --resolution 4
"""


def run() -> None:
    parser = argparse.ArgumentParser(description=RUN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. dense.method=colmap")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Pipeline(load_config(args.config, overrides)).run(args.stages)


def download() -> None:
    parser = argparse.ArgumentParser(description=DOWNLOAD_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", nargs="+", default=["counter"], choices=[*SCENES, "indoor", "all"])
    parser.add_argument("--resolution", type=int, default=4, choices=RESOLUTIONS,
                        help="downsampling factor of the images (4 = ~780x520)")
    parser.add_argument("--out", default="data/mipnerf360")
    args = parser.parse_args()

    scenes = []
    for s in args.scene:
        scenes += list(SCENES) if s == "all" else list(INDOOR_SCENES) if s == "indoor" else [s]
    for scene in dict.fromkeys(scenes):
        path = download_scene(scene, args.out, args.resolution)
        print(f"{scene}: {path}")

