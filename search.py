"""Captioning, embeddings and the search index.

Pipeline:   image --(vision LLM)--> caption --(embedding model)--> vector --> data/index.json
Query:      text --(embedding model)--> vector --> cosine similarity vs. every stored vector

Semantic search needs an embedding API key (Gemini's embedding model has a free tier). With no
key the index still works using keyword (TF-IDF) matching, so the app never loses search.
Everything is stored in one JSON file: simple, inspectable, and plenty for a personal library.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from providers import EditError, GeminiEditor, OpenAIEditor, _env

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- retry
def _retry(fn, tries: int = 5, base: float = 3.0):
    """Retry transient API errors (rate limits, 503). Hard quota errors fail immediately."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(t in msg for t in ("429", "503", "unavailable", "timeout", "timed out"))
            if not transient or "limit: 0" in msg or i == tries - 1:
                raise
            time.sleep(base * (2 ** i))


# ---------------------------------------------------------------- captioning
class Captioner:
    def __init__(self, backend=None, name: str = "none"):
        self.backend = backend
        self.name = name

    @property
    def available(self) -> bool:
        return self.backend is not None

    def caption(self, img) -> str:
        if not self.backend:
            raise EditError("No caption provider configured (set GEMINI_API_KEY or OPENAI_API_KEY).")
        text = (_retry(lambda: self.backend.caption(img)) or "").strip()
        if not text:
            raise EditError("The model returned an empty caption.")
        return text


def get_captioner() -> Captioner:
    pref = _env("CAPTION_PROVIDER", "auto").lower()
    order = ["gemini", "openai"] if pref == "auto" else [pref]
    for p in order:
        try:
            if p == "gemini" and (_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")):
                return Captioner(GeminiEditor(), "gemini")
            if p == "openai" and _env("OPENAI_API_KEY"):
                return Captioner(OpenAIEditor(), "openai")
        except Exception as e:
            log.warning("Captioner '%s' unavailable: %s", p, e)
    return Captioner()


# ---------------------------------------------------------------- embeddings
class Embedder:
    name = "base"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GeminiEmbedder(Embedder):
    def __init__(self):
        from google import genai
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if not key:
            raise EditError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)
        self.model = _env("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.name = f"gemini:{self.model}"

    def _embed(self, texts, task):
        from google.genai import types
        r = _retry(lambda: self.client.models.embed_content(
            model=self.model, contents=texts,
            config=types.EmbedContentConfig(task_type=task)))
        return [list(e.values) for e in r.embeddings]

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OpenAIEmbedder(Embedder):
    def __init__(self):
        from openai import OpenAI
        if not _env("OPENAI_API_KEY"):
            raise EditError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()
        self.model = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.name = f"openai:{self.model}"

    def embed_documents(self, texts):
        r = _retry(lambda: self.client.embeddings.create(model=self.model, input=texts))
        return [d.embedding for d in r.data]


def get_embedder() -> Embedder | None:
    pref = _env("EMBED_PROVIDER", "auto").lower()
    order = ["gemini", "openai"] if pref == "auto" else [pref]
    for p in order:
        try:
            if p == "gemini" and (_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")):
                return GeminiEmbedder()
            if p == "openai" and _env("OPENAI_API_KEY"):
                return OpenAIEmbedder()
        except Exception as e:
            log.warning("Embedder '%s' unavailable: %s", p, e)
    return None


# ---------------------------------------------------------------- text helpers
_STOP = set("a an the of and or in on at to with is are was for from by it its this that as into over under".split())


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 1]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _doc_text(caption: str, name: str) -> str:
    stem = re.sub(r"[_\-.]+", " ", os.path.splitext(name or "")[0]).strip()
    return f"{stem}. {caption}" if stem else caption


# ---------------------------------------------------------------- the index
class SearchIndex:
    SCHEMA = 1

    def __init__(self, path: str | Path | None = None, embedder: Embedder | None = None):
        self.path = Path(path or Path(os.getenv("DATA_DIR", "data")) / "index.json")
        self.embedder = embedder
        self.items: dict[str, dict] = {}
        self.last_error: str | None = None
        self._load()

    # --- persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.items = json.loads(self.path.read_text(encoding="utf-8")).get("items", {})
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Search index unreadable (%s); starting fresh.", e)
            try:
                self.path.rename(self.path.with_suffix(".corrupt"))
            except OSError:
                pass
            self.items = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema": self.SCHEMA, "items": self.items}, f)
        os.replace(tmp, self.path)

    # --- maintenance
    def upsert(self, image_id: str, caption: str, name: str = "") -> bool:
        """Add/update an entry. Returns True if an embedding was stored."""
        item = {"caption": caption, "name": name, "embedding": None, "embed_model": None,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if self.embedder and caption:
            try:
                item["embedding"] = self.embedder.embed_documents([_doc_text(caption, name)])[0]
                item["embed_model"] = self.embedder.name
                self.last_error = None
            except Exception as e:
                self.last_error = f"Embedding failed, using keyword search for this image: {str(e)[:160]}"
                log.warning(self.last_error)
        self.items[image_id] = item
        self.save()
        return item["embedding"] is not None

    def remove(self, image_id: str) -> None:
        if self.items.pop(image_id, None) is not None:
            self.save()

    def sync(self, valid_ids: set[str]) -> int:
        """Drop entries for images that no longer exist. Returns how many were removed."""
        stale = [i for i in self.items if i not in valid_ids]
        for i in stale:
            del self.items[i]
        if stale:
            self.save()
        return len(stale)

    def needs_embedding(self, image_id: str) -> bool:
        it = self.items.get(image_id)
        return bool(self.embedder) and (not it or it.get("embed_model") != self.embedder.name)

    def stats(self) -> dict:
        embedded = sum(1 for it in self.items.values() if it.get("embedding"))
        return {"indexed": len(self.items), "embedded": embedded,
                "mode": "semantic" if self.embedder else "keyword",
                "model": self.embedder.name if self.embedder else None}

    # --- search
    def _lexical_scores(self, query: str) -> dict[str, float]:
        docs = {i: Counter(tokenize(_doc_text(it.get("caption", ""), it.get("name", ""))))
                for i, it in self.items.items()}
        n = len(docs)
        df = Counter(t for c in docs.values() for t in c)
        idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        q = Counter(tokenize(query))
        qv = {t: c * idf.get(t, math.log(n + 1) + 1) for t, c in q.items()}
        qn = math.sqrt(sum(v * v for v in qv.values()))
        scores = {}
        for i, c in docs.items():
            dv = {t: cnt * idf[t] for t, cnt in c.items()}
            dn = math.sqrt(sum(v * v for v in dv.values()))
            dot = sum(qv[t] * dv.get(t, 0.0) for t in qv)
            scores[i] = dot / (qn * dn) if qn and dn else 0.0
        return scores

    def search(self, query: str, k: int = 24) -> tuple[list[tuple[str, float]], str]:
        """Return ([(image_id, score)...] best first, mode) where mode is 'semantic' or 'keyword'."""
        if not query.strip() or not self.items:
            return [], "keyword"
        lex = self._lexical_scores(query)
        scores, mode = dict(lex), "keyword"
        if self.embedder:
            usable = {i: it for i, it in self.items.items()
                      if it.get("embedding") and it.get("embed_model") == self.embedder.name}
            if usable:
                try:
                    qv = self.embedder.embed_query(query)
                    for i, it in usable.items():
                        scores[i] = 0.85 * max(0.0, cosine(qv, it["embedding"])) + 0.15 * lex.get(i, 0.0)
                    mode = "semantic"
                    self.last_error = None
                except Exception as e:
                    self.last_error = f"Semantic search unavailable, used keyword matching: {str(e)[:160]}"
                    log.warning(self.last_error)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return ranked, mode