"""AI-Powered Image Editing Platform - Week 3: library search + backend enhancements.
Run:  streamlit run app.py
"""
import io
import logging
import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import prompts
from providers import EditError, get_editor
from search import SearchIndex, get_captioner, get_embedder
from storage import ImageStore, StorageError

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="AI Image Editor", page_icon="🎨", layout="wide")

store = ImageStore()
captioner = get_captioner()
index = SearchIndex(embedder=get_embedder())

ss = st.session_state
ss.setdefault("selected", None)       # image_id open in the detail view
ss.setdefault("viewing", None)        # version id shown in the main pane
ss.setdefault("pending_prompt", "")
ss.setdefault("uploader_key", 0)
ss.setdefault("flash", [])            # messages shown once after a rerun


def fmt_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")


@st.cache_resource
def editor_for(provider: str):
    return get_editor(provider)


def filename_caption(name: str) -> str:
    base = re.sub(r"[_\-]+", " ", os.path.splitext(name)[0]).strip()
    return base or "untitled image"


def refresh_caption(image_id: str, img, name: str) -> tuple[bool, str]:
    """Caption an image with the vision model, save it and (re)index it."""
    try:
        text = captioner.caption(img)
    except Exception as e:
        return False, f"Captioning failed: {str(e)[:200]}"
    store.set_caption(image_id, text, "ai")
    index.upsert(image_id, text, name)
    return True, ""


def ensure_indexed(meta: dict) -> None:
    if meta.get("caption"):
        index.upsert(meta["id"], meta["caption"], meta["name"])


def card_caption(meta: dict, limit: int = 90) -> str:
    cap = meta.get("caption") or "(no caption)"
    return cap if len(cap) <= limit else cap[:limit].rsplit(" ", 1)[0] + "…"


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🎨 AI Image Editor")
    options = ["hfspace", "huggingface", "gemini", "openai", "mock"]
    default = os.getenv("PROVIDER", "mock")
    provider = st.selectbox("Image-edit provider", options,
                            index=options.index(default) if default in options else options.index("mock"),
                            help="'hfspace' and 'huggingface' are free (token needed). 'mock' applies simple filters.")
    use_planner = st.toggle("Smart prompt interpretation", value=True,
                            help="A vision LLM first rewrites your request into a precise edit spec.")
    auto_caption = st.toggle("Re-caption after edits", value=True, disabled=not captioner.available,
                             help="Keeps search results accurate after an image changes.")
    st.divider()

    up = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "webp"],
                          accept_multiple_files=True, key=f"uploader_{ss.uploader_key}")
    if up and st.button("Add to library", width="stretch"):
        notes, added = [], 0
        with st.spinner("Adding and captioning..."):
            for f in up:
                data = f.getvalue()
                if store.find_duplicate(data):
                    notes.append(f"{f.name}: already in your library, skipped.")
                    continue
                try:
                    image_id = store.add_image(data, f.name)
                except StorageError as e:
                    notes.append(str(e))
                    continue
                ok, msg = refresh_caption(image_id, store.load_version(image_id, 0), f.name)
                if not ok:  # keep the image searchable by its file name
                    store.set_caption(image_id, filename_caption(f.name), "filename")
                    index.upsert(image_id, filename_caption(f.name), f.name)
                    notes.append(f"{f.name}: {msg} Using the file name as its caption; you can edit it.")
                added += 1
        if added:
            notes.insert(0, f"Added {added} image(s).")
        ss.flash, ss.selected, ss.viewing = notes, None, None
        ss.uploader_key += 1
        st.rerun()

    if ss.selected and st.button("← Back to library", width="stretch"):
        ss.selected = None
        st.rerun()

    st.divider()
    stats = index.stats()
    st.caption(f"Captions: **{captioner.name}** · Search: **{stats['mode']}**"
               + (f" ({stats['model']})" if stats["model"] else ""))
    st.caption(f"Indexed {stats['indexed']} · with embeddings {stats['embedded']}")
    if not captioner.available:
        st.info("Add GEMINI_API_KEY (free) to .env for AI captions and semantic search.")
    if st.button("Rebuild search index", width="stretch",
                 help="Re-captions images that only have a file-name caption and re-embeds anything out of date."):
        metas = store.list_images()
        index.sync({m["id"] for m in metas})
        bar, problems = st.progress(0.0), []
        for n, m in enumerate(metas, 1):
            try:
                if captioner.available and m.get("caption_source") in (None, "filename"):
                    latest = store.load_version(m["id"], store.latest_version_id(m["id"]))
                    ok, msg = refresh_caption(m["id"], latest, m["name"])
                    if not ok:
                        problems.append(f"{m['name']}: {msg}")
                        ensure_indexed(m)
                elif index.needs_embedding(m["id"]) or m["id"] not in index.items:
                    ensure_indexed(m)
            except StorageError as e:
                problems.append(f"{m['name']}: {e}")
            bar.progress(n / len(metas))
        ss.flash = ["Search index rebuilt."] + problems
        st.rerun()

# ---------------- flash messages ----------------
for msg in ss.flash:
    (st.success if msg.startswith(("Added", "Search index", "Image deleted")) else st.warning)(msg)
ss.flash = []
if index.last_error:
    st.warning(index.last_error)

# ---------------- library + search ----------------
if not ss.selected:
    st.header("Library")
    metas = store.list_images()
    for m in metas:                      # make sure every captioned image is in the index
        if m["id"] not in index.items:
            ensure_indexed(m)
    index.sync({m["id"] for m in metas})

    query = st.text_input("🔍 Search your images", placeholder='Try "sunset", "a tiger", "beach with people"',
                          label_visibility="collapsed")
    shown = metas
    scores: dict[str, float] = {}
    if query.strip():
        ranked, mode = index.search(query.strip(), k=len(metas))
        default_min = 0.45 if mode == "semantic" else 0.05
        c1, c2 = st.columns([3, 1])
        min_score = c2.slider("Min relevance", 0.0, 1.0, default_min, 0.05,
                              help="Raise to see only closer matches.")
        by_id = {m["id"]: m for m in metas}
        shown = [by_id[i] for i, s in ranked if s >= min_score and i in by_id]
        scores = dict(ranked)
        c1.caption(f"{len(shown)} match(es) · {mode} search")
        if not shown:
            st.info("No images matched. Lower 'Min relevance' or try different words.")
    elif not metas:
        st.info("Upload an image from the sidebar to get started.")

    cols = st.columns(4)
    for i, meta in enumerate(shown):
        with cols[i % 4]:
            st.image(str(store.thumb_path(meta["id"], store.latest_version_id(meta["id"]))), width="stretch")
            st.markdown(f"**{meta['name']}**")
            st.caption(card_caption(meta))
            st.caption(f"{len(meta['versions'])} version(s)" +
                       (f" · relevance {scores[meta['id']]:.2f}" if meta["id"] in scores else ""))
            if st.button("Open", key=f"open_{meta['id']}", width="stretch"):
                ss.selected, ss.viewing = meta["id"], None
                st.rerun()
    st.stop()

# ---------------- detail + edit ----------------
try:
    meta = store.get(ss.selected)
    latest = store.latest_version_id(ss.selected)
    viewing = latest if ss.viewing is None else ss.viewing
    current_img = store.load_version(ss.selected, viewing)
except StorageError as e:
    st.error(str(e))
    ss.selected = None
    st.stop()

st.header(meta["name"])
left, right = st.columns([3, 2], gap="large")

with left:
    tab_cur, tab_cmp = st.tabs([f"Version v{viewing}", "Before / After"])
    with tab_cur:
        st.image(current_img, width="stretch")
    with tab_cmp:
        a, b = st.columns(2)
        a.image(store.load_version(ss.selected, 0), caption="Original (v0)", width="stretch")
        b.image(current_img, caption=f"v{viewing}", width="stretch")

    with st.expander("Caption & search", expanded=False):
        source = meta.get("caption_source") or "none"
        st.caption(f"This caption is what search matches against (source: {source}).")
        new_cap = st.text_area("Caption", value=meta.get("caption") or "", height=90,
                               key=f"cap_{ss.selected}_{abs(hash(meta.get('caption') or ''))}")
        c1, c2 = st.columns(2)
        if c1.button("Save caption", width="stretch") and new_cap.strip():
            store.set_caption(ss.selected, new_cap, "manual")
            index.upsert(ss.selected, new_cap.strip(), meta["name"])
            st.rerun()
        if c2.button("Regenerate with AI", width="stretch", disabled=not captioner.available):
            with st.spinner("Captioning..."):
                ok, msg = refresh_caption(ss.selected, current_img, meta["name"])
            if not ok:
                ss.flash = [msg]
            st.rerun()

with right:
    st.subheader("Edit")
    st.caption(f"Editing from **v{viewing}**. The result is saved as a new version.")

    st.write("Quick edits")
    pcols = st.columns(2)
    for i, (label, text) in enumerate(prompts.PRESETS.items()):
        if pcols[i % 2].button(label, key=f"preset_{i}", width="stretch"):
            ss.pending_prompt = text

    request = st.text_area("Or describe your edit", value=ss.pending_prompt, height=100,
                           placeholder='e.g. "remove the person on the left" or "add a tree in the background"')
    if st.button("✨ Apply edit", type="primary", width="stretch", disabled=not request.strip()):
        try:
            editor = editor_for(provider)
            with st.spinner("Editing with AI... this can take 10-60 seconds"):
                new_img, edit_prompt, spec = editor.edit(current_img, request.strip(), use_planner)
                v = store.add_version(ss.selected, new_img, prompt=request.strip(), kind="edit",
                                      provider=editor.name, parent=viewing)
            ss.viewing, ss.pending_prompt = v["id"], ""
            ss.last_debug = {"spec": spec, "edit_prompt": edit_prompt}
            if auto_caption and captioner.available:
                ok, msg = refresh_caption(ss.selected, new_img, meta["name"])
                if not ok:
                    ss.flash = [msg]
            st.rerun()
        except EditError as e:
            st.error(str(e))
        except Exception as e:  # network/quota/etc.
            st.error(f"Edit failed: {e}")

    if ss.get("last_debug"):
        with st.expander("How was my request interpreted?"):
            st.json(ss.last_debug["spec"] or {"note": "planner skipped"})
            st.code(ss.last_debug["edit_prompt"], language="text")

    buf = io.BytesIO()
    current_img.save(buf, format="PNG")
    st.download_button(f"Download v{viewing}", buf.getvalue(),
                       f"{meta['name'].rsplit('.', 1)[0]}_v{viewing}.png", "image/png", width="stretch")

    with st.expander("Danger zone"):
        confirm = st.checkbox("I want to delete this image and all its versions")
        if st.button("Delete image", disabled=not confirm, width="stretch"):
            store.delete_image(ss.selected)
            index.remove(ss.selected)
            ss.selected, ss.viewing = None, None
            ss.flash = ["Image deleted."]
            st.rerun()

# ---------------- version history ----------------
st.divider()
st.subheader("Version history")
versions = list(reversed(meta["versions"]))
hcols = st.columns(5)
for i, v in enumerate(versions):
    with hcols[i % 5]:
        st.image(str(store.thumb_path(ss.selected, v["id"])), width="stretch")
        tag = "Original" if v["kind"] == "original" else f"v{v['id']}"
        st.markdown(f"**{tag}**" + (" · viewing" if v["id"] == viewing else ""))
        st.caption(fmt_time(v["created_at"]))
        if v["kind"] != "original":
            st.caption(v["prompt"][:80])
        c1, c2 = st.columns(2)
        if c1.button("View", key=f"view_{v['id']}"):
            ss.viewing = v["id"]
            st.rerun()
        if v["id"] != latest and c2.button("Revert", key=f"rev_{v['id']}",
                                           help="Copies this version forward as the newest"):
            nv = store.revert_to(ss.selected, v["id"])
            ss.viewing = nv["id"]
            if auto_caption and captioner.available:
                refresh_caption(ss.selected, store.load_version(ss.selected, nv["id"]), meta["name"])
            st.rerun()