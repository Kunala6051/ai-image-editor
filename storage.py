"""Local image storage + non-destructive version tracking (schema v2).

Layout on disk:
  data/images/<image_id>/meta.json
  data/images/<image_id>/v0.png, v1.png, ...        full-size versions (v0 = untouched original)
  data/images/<image_id>/thumbs/v0.png, ...         thumbnails
  data/index.json                                   search index (see search.py)

Improvements over Week 2: atomic JSON writes (no half-written files), schema versioning with
automatic upgrade of old metadata, duplicate-upload detection (sha256), validation of uploads,
and skipping (not crashing on) corrupt entries.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

THUMB_SIZE = (256, 256)
MAX_SIDE = 4096          # larger uploads are downscaled to keep the app fast
SCHEMA_VERSION = 2


class StorageError(Exception):
    """User-facing storage problem (invalid image, missing image, ...)."""


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
        _atomic_write_json(self._meta_path(image_id), meta)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _upgrade(meta: dict) -> dict:
        """Fill in fields added in newer schema versions so old data keeps working."""
        meta.setdefault("schema", 1)
        meta.setdefault("sha256", None)
        meta.setdefault("caption", None)
        meta.setdefault("caption_source", None)   # "ai" | "manual" | "filename"
        meta.setdefault("caption_updated", None)
        meta.setdefault("versions", [])
        return meta

    # ---------- images ----------
    def add_image(self, data: bytes, name: str) -> str:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise StorageError(f"'{name}' is not a valid image file.") from e
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        image_id = uuid.uuid4().hex[:10]
        (self._dir(image_id) / "thumbs").mkdir(parents=True)
        meta = {"schema": SCHEMA_VERSION, "id": image_id, "name": name,
                "created_at": self._now(), "sha256": hashlib.sha256(data).hexdigest(),
                "caption": None, "caption_source": None, "caption_updated": None,
                "versions": []}
        self._write_meta(image_id, meta)
        self.add_version(image_id, img, prompt="(original upload)",
                         kind="original", provider="-", parent=None)
        return image_id

    def find_duplicate(self, data: bytes) -> str | None:
        digest = hashlib.sha256(data).hexdigest()
        for m in self.list_images():
            if m.get("sha256") == digest:
                return m["id"]
        return None

    def list_images(self) -> list[dict]:
        metas = []
        for p in self.root.glob("*/meta.json"):
            try:
                metas.append(self._upgrade(json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Skipping unreadable metadata %s: %s", p, e)
        return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)

    def get(self, image_id: str) -> dict:
        try:
            return self._upgrade(json.loads(self._meta_path(image_id).read_text(encoding="utf-8")))
        except FileNotFoundError as e:
            raise StorageError("That image no longer exists.") from e
        except json.JSONDecodeError as e:
            raise StorageError("This image's metadata is corrupted.") from e

    def set_caption(self, image_id: str, caption: str, source: str) -> None:
        meta = self.get(image_id)
        meta["caption"] = caption.strip()
        meta["caption_source"] = source
        meta["caption_updated"] = self._now()
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
        path = self._dir(image_id) / f"v{version_id}.png"
        if not path.exists():
            raise StorageError(f"Version v{version_id} is missing on disk.")
        return Image.open(path).convert("RGB")

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