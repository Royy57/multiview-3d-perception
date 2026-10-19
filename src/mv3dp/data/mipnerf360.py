"""Download a single Mip-NeRF 360 scene without fetching the whole 12.5 GB archive.

The official release is one zip on Google Cloud Storage. Zip archives keep a
central directory at the end, so with HTTP range requests we can read that
directory and then pull only the members of one scene at one resolution.
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

MIPNERF360_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENES = ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump")
INDOOR_SCENES = ("bonsai", "counter", "kitchen", "room")
RESOLUTIONS = (1, 2, 4, 8)


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP using Range requests."""

    def __init__(self, url: str, block_size: int = 1 << 20):
        self.url = url
        self.pos = 0
        self.block_size = block_size
        self._cache_start = -1
        self._cache = b""
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        if head.headers.get("Accept-Ranges", "bytes") != "bytes":
            raise RuntimeError(f"{url} does not support range requests")
        self.size = int(head.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n
        cache_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= start and end <= cache_end):
            # Small reads (zip headers) are served from a read-ahead block.
            fetch_end = min(self.size, max(end, start + self.block_size)) if n < self.block_size else end
            self._cache = self._fetch(start, fetch_end)
            self._cache_start = start
        offset = start - self._cache_start
        data = self._cache[offset : offset + n]
        self.pos += len(data)
        return data


def download_scene(
    scene: str,
    out_dir: str | Path,
    resolution: int = 4,
    url: str = MIPNERF360_URL,
) -> Path:
    """Fetch ``<scene>/images_<resolution>`` plus the reference COLMAP model.

    Images are written to ``out_dir/<scene>/images`` so downstream code does not
    depend on the downsampling factor. Returns the scene directory.
    """
    if scene not in SCENES:
        raise ValueError(f"Unknown scene '{scene}'. Choose from {SCENES}.")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}.")

    scene_dir = Path(out_dir) / scene
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_prefix = f"{scene}/images_{resolution}/" if resolution > 1 else f"{scene}/images/"
    sparse_prefix = f"{scene}/sparse/0/"

    with zipfile.ZipFile(HTTPRangeFile(url)) as archive:
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and (info.filename.startswith(image_prefix) or info.filename.startswith(sparse_prefix))
        ]
        total = sum(info.file_size for info in members)
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"{scene} x1/{resolution}") as bar:
            for info in members:
                if info.filename.startswith(image_prefix):
                    target = image_dir / Path(info.filename).name
                else:
                    target = scene_dir / "reference_sparse" / Path(info.filename).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == info.file_size:
                    bar.update(info.file_size)
                    continue
                with archive.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                bar.update(info.file_size)

    (scene_dir / "resolution.txt").write_text(f"{resolution}\n")
    return scene_dir
