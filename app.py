"""AI-Powered Image Editing Platform - Week 2: AI editing + version history.
Run:  streamlit run app.py
"""
import io
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import prompts
from providers import EditError, get_editor
from storage import ImageStore

load_dotenv()
st.set_page_config(page_title="AI Image Editor", page_icon="🎨", layout="wide")
store = ImageStore()

ss = st.session_state
ss.setdefault("selected", None)      # image_id being edited
ss.setdefault("viewing", None)       # version id shown in the main pane
ss.setdefault("pending_prompt", "")


def fmt_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")


@st.cache_resource
def editor_for(provider: str):
    return get_editor(provider)


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🎨 AI Image Editor")
    options = ["hfspace", "huggingface", "gemini", "openai", "mock"]
    default = os.getenv("PROVIDER", "mock")
    provider = st.selectbox("AI provider", options,
                            index=options.index(default) if default in options else options.index("mock"),
                            help="'huggingface' is free (token needed). 'mock' applies simple filters with no key.")
    use_planner = st.toggle("Smart prompt interpretation", value=True,
                            help="A vision LLM first rewrites your request into a precise edit spec.")
    st.divider()
    up = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if up and st.button("Add to library", width="stretch"):
        for f in up:
            ss.selected = store.add_image(f.getvalue(), f.name)
        ss.viewing = None
        st.rerun()
    if ss.selected and st.button("← Back to library", width="stretch"):
        ss.selected = None
        st.rerun()

# ---------------- library ----------------
if not ss.selected:
    st.header("Library")
    images = store.list_images()
    if not images:
        st.info("Upload an image from the sidebar to get started.")
    cols = st.columns(4)
    for i, meta in enumerate(images):
        with cols[i % 4]:
            st.image(str(store.thumb_path(meta["id"], store.latest_version_id(meta["id"]))),
                     width="stretch")
            st.caption(f"{meta['name']} · {len(meta['versions'])} version(s)")
            if st.button("Open", key=f"open_{meta['id']}", width="stretch"):
                ss.selected, ss.viewing = meta["id"], None
                st.rerun()
    st.stop()

# ---------------- detail + edit ----------------
meta = store.get(ss.selected)
latest = store.latest_version_id(ss.selected)
viewing = latest if ss.viewing is None else ss.viewing
current_img = store.load_version(ss.selected, viewing)

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
            with st.spinner("Editing with AI... this can take 10-30 seconds"):
                new_img, edit_prompt, spec = editor.edit(current_img, request.strip(), use_planner)
                v = store.add_version(ss.selected, new_img, prompt=request.strip(), kind="edit",
                                      provider=editor.name, parent=viewing)
            ss.viewing, ss.pending_prompt = v["id"], ""
            ss.last_debug = {"spec": spec, "edit_prompt": edit_prompt}
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
    st.download_button(f"Download v{viewing}", buf.getvalue(), f"{meta['name'].rsplit('.', 1)[0]}_v{viewing}.png",
                       "image/png", width="stretch")

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
        if v["id"] != latest and c2.button("Revert", key=f"rev_{v['id']}", help="Copies this version forward as the newest"):
            nv = store.revert_to(ss.selected, v["id"])
            ss.viewing = nv["id"]
            st.rerun()
