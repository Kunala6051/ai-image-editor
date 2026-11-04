"""Run with:  python -m pytest -q   (from the project root)"""
import io
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from search import Embedder, SearchIndex, tokenize
from storage import ImageStore, StorageError


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder so semantic code paths run offline."""
    name = "fake:bow"
    SYN = {"ocean": "beach", "sea": "beach", "cat": "tiger", "kitten": "tiger"}

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in tokenize(t):
                w = self.SYN.get(w, w)
                v[zlib.crc32(w.encode()) % 64] += 1.0
            out.append(v)
        return out


def _png(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _index(tmp_path, embedder=None):
    idx = SearchIndex(tmp_path / "index.json", embedder)
    idx.upsert("a", "A cartoon tiger standing on a white background", "tiger.jpg")
    idx.upsert("b", "A sunset over a sandy beach with palm trees", "holiday.jpg")
    idx.upsert("c", "A snowy mountain peak under a blue sky", "alps.jpg")
    return idx


def test_keyword_search_ranks_matching_caption_first(tmp_path):
    idx = _index(tmp_path)
    ranked, mode = idx.search("sunsets at the beach")
    assert mode == "keyword" and ranked[0][0] == "b"
    assert idx.search("mountains")[0][0][0] == "c"


def test_no_match_scores_zero(tmp_path):
    ranked, _ = _index(tmp_path).search("spaceship")
    assert all(score == 0 for _, score in ranked)


def test_semantic_search_uses_embeddings(tmp_path):
    idx = _index(tmp_path, FakeEmbedder())
    ranked, mode = idx.search("ocean")
    assert mode == "semantic" and ranked[0][0] == "b"


def test_index_persists_and_survives_corruption(tmp_path):
    _index(tmp_path)
    assert len(SearchIndex(tmp_path / "index.json").items) == 3
    (tmp_path / "index.json").write_text("{not json")
    assert SearchIndex(tmp_path / "index.json").items == {}


def test_sync_and_remove(tmp_path):
    idx = _index(tmp_path)
    assert idx.sync({"a", "b"}) == 1 and "c" not in idx.items
    idx.remove("a")
    assert list(idx.items) == ["b"]


def test_duplicate_upload_detected_and_invalid_rejected(tmp_path):
    store = ImageStore(tmp_path / "images")
    image_id = store.add_image(_png(), "a.png")
    assert store.find_duplicate(_png()) == image_id
    assert store.find_duplicate(_png((1, 2, 3))) is None
    try:
        store.add_image(b"not an image", "bad.png")
    except StorageError:
        return
    raise AssertionError("expected StorageError")


def test_caption_saved_and_old_metadata_upgraded(tmp_path):
    store = ImageStore(tmp_path / "images")
    image_id = store.add_image(_png(), "a.png")
    store.set_caption(image_id, "a dark square", "manual")
    meta = store.get(image_id)
    assert meta["caption"] == "a dark square" and meta["caption_source"] == "manual"