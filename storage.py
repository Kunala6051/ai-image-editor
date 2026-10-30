"""Local image storage + non-destructive version tracking.

Layout on disk:
  data/images/<image_id>/meta.json
  data/images/<image_id>/v0.png, v1.png, ...        (full-size versions)
  data/images/<image_id>/thumbs/v0.png, v1.png ...  (thumbnails)

v0 is always the untouched original. Every edit (or revert) adds a new
version that records its parent, the prompt used, the provider and a timestamp.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

THUMB_SIZE = (256, 256)


class ImageStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or Path(os.getenv("DATA_DIR", "data")) / "images")
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _dir(self, image_id: str) -> Path:
        return self.root / image_id

    def _meta_path(self, image_id: str) -> Path:
        return self._dir(image_id) / "meta.json"

    def _write_meta(self, image_id: str, meta: dict) -> None:
        self._meta_path(image_id).write_text(json.dumps(meta, indent=2))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------- images ----------
    def add_image(self, data: bytes, name: str) -> str:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        image_id = uuid.uuid4().hex[:10]
        (self._dir(image_id) / "thumbs").mkdir(parents=True)
        meta = {"id": image_id, "name": name, "created_at": self._now(),
                "caption": None, "versions": []}
        self._write_meta(image_id, meta)
        self.add_version(image_id, img, prompt="(original upload)",
                         kind="original", provider="-", parent=None)
        return image_id

    def list_images(self) -> list[dict]:
        metas = [json.loads(p.read_text()) for p in self.root.glob("*/meta.json")]
        return sorted(metas, key=lambda m: m["created_at"], reverse=True)

    def get(self, image_id: str) -> dict:
        return json.loads(self._meta_path(image_id).read_text())

    def set_caption(self, image_id: str, caption: str) -> None:
        meta = self.get(image_id)
        meta["caption"] = caption
        self._write_meta(image_id, meta)

    def delete_image(self, image_id: str) -> None:
        shutil.rmtree(self._dir(image_id), ignore_errors=True)

    # ---------- versions ----------
    def add_version(self, image_id: str, img: Image.Image, prompt: str, kind: str,
                    provider: str, parent: int | None) -> dict:
        meta = self.get(image_id)
        vid = len(meta["versions"])
        img = img.convert("RGB")
        img.save(self._dir(image_id) / f"v{vid}.png")
        thumb = img.copy()
        thumb.thumbnail(THUMB_SIZE)
        thumb.save(self._dir(image_id) / "thumbs" / f"v{vid}.png")
        version = {"id": vid, "parent": parent, "prompt": prompt, "kind": kind,
                   "provider": provider, "created_at": self._now(),
                   "size": list(img.size)}
        meta["versions"].append(version)
        self._write_meta(image_id, meta)
        return version

    def load_version(self, image_id: str, version_id: int) -> Image.Image:
        return Image.open(self._dir(image_id) / f"v{version_id}.png").convert("RGB")

    def thumb_path(self, image_id: str, version_id: int) -> Path:
        return self._dir(image_id) / "thumbs" / f"v{version_id}.png"

    def revert_to(self, image_id: str, version_id: int) -> dict:
        """Non-destructive revert: copies an old version forward as a new one."""
        img = self.load_version(image_id, version_id)
        latest = len(self.get(image_id)["versions"]) - 1
        return self.add_version(image_id, img, prompt=f"(reverted to v{version_id})",
                                kind="revert", provider="-", parent=latest)

    def latest_version_id(self, image_id: str) -> int:
        return len(self.get(image_id)["versions"]) - 1
