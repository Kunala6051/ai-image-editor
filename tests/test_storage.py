"""Run with:  python -m pytest -q   (from the project root)"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from providers import EditError, MockEditor, get_editor
from storage import ImageStore


def _png(color=(200, 120, 60), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_creates_original_version(tmp_path):
    store = ImageStore(tmp_path)
    image_id = store.add_image(_png(), "a.png")
    meta = store.get(image_id)
    assert len(meta["versions"]) == 1
    assert meta["versions"][0]["kind"] == "original"
    assert store.thumb_path(image_id, 0).exists()


def test_edit_adds_version_with_parent_and_prompt(tmp_path):
    store = ImageStore(tmp_path)
    image_id = store.add_image(_png(), "a.png")
    out, _, _ = MockEditor().edit(store.load_version(image_id, 0), "make it black and white")
    v = store.add_version(image_id, out, "make it black and white", "edit", "mock", parent=0)
    assert v["id"] == 1 and v["parent"] == 0
    assert store.get(image_id)["versions"][1]["prompt"] == "make it black and white"


def test_revert_is_non_destructive(tmp_path):
    store = ImageStore(tmp_path)
    image_id = store.add_image(_png(), "a.png")
    store.add_version(image_id, Image.new("RGB", (64, 48), (0, 0, 0)), "dark", "edit", "mock", 0)
    store.revert_to(image_id, 0)
    versions = store.get(image_id)["versions"]
    assert [v["kind"] for v in versions] == ["original", "edit", "revert"]
    assert store.load_version(image_id, 2).getpixel((0, 0)) == store.load_version(image_id, 0).getpixel((0, 0))


def test_unknown_provider_raises():
    try:
        get_editor("nope")
    except EditError:
        return
    raise AssertionError("expected EditError")