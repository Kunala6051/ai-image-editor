"""Thin adapters around the GenAI APIs. The app only talks to `get_editor()`."""
from __future__ import annotations

import base64
import io
import json
import os
import re

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import prompts

def _env(name: str, default: str = "") -> str:
    """Read an env var, ignoring inline '# comments'."""
    v = os.getenv(name, default) or ""
    v = v.split(" #")[0].strip()
    return "" if v.startswith("#") else v


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else None


class EditError(Exception):
    """User-facing error (bad key, refusal, infeasible request...)."""


def _friendly(e: Exception, model: str) -> EditError:
    msg = str(e)
    low = msg.lower()
    if "resource_exhausted" in low or "429" in low or "quota" in low:
        return EditError(f"Quota/rate limit hit for {model}. This model may not be on your free tier. "
                         "Try PROVIDER=huggingface (free), wait a minute, or use mock.")
    if "401" in low or "403" in low or "api key" in low or "unauthorized" in low or "invalid" in low and "token" in low:
        return EditError(f"Authentication failed for {model}. Check your API key/token in .env.")
    if "402" in low or "credit" in low or "payment" in low:
        return EditError(f"Free credits used up for {model}. Switch to PROVIDER=hfspace (uses free Space GPU quota) or wait for the monthly reset.")
    if "503" in low or "loading" in low or "timeout" in low or "timed out" in low:
        return EditError(f"{model} is busy or warming up. Wait ~30s and try again.")
    return EditError(f"{model} failed: {msg[:300]}")


class BaseEditor:
    name = "base"

    def interpret(self, img: Image.Image, request: str) -> dict | None:
        return None  # no planner -> fall back to SIMPLE_TEMPLATE

    def caption(self, img: Image.Image) -> str:
        return ""

    def generate(self, img: Image.Image, edit_prompt: str) -> Image.Image:
        raise NotImplementedError

    def edit(self, img: Image.Image, request: str, use_planner: bool = True):
        """Full pipeline. Returns (new_image, edit_prompt_used, spec)."""
        spec = None
        if use_planner:
            try:
                spec = self.interpret(img, request)
            except Exception:
                spec = None  # planner is best-effort; never block the edit
        if spec and spec.get("feasible") is False:
            raise EditError(spec.get("reason_if_infeasible") or "That edit doesn't seem possible for this image.")
        edit_prompt = prompts.build_edit_prompt(spec, request)
        return self.generate(img, edit_prompt), edit_prompt, spec


class GeminiEditor(BaseEditor):
    name = "gemini"

    def __init__(self):
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EditError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)
        self.image_model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        self.text_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

    def interpret(self, img, request):
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.text_model,
            contents=[img, f"User request: {request}"],
            config=types.GenerateContentConfig(
                system_instruction=prompts.INTERPRETER_SYSTEM, temperature=0.1,
                response_mime_type="application/json"),
        )
        return _parse_json(resp.text or "")

    def caption(self, img):
        resp = self.client.models.generate_content(
            model=self.text_model, contents=[img, prompts.CAPTION_PROMPT])
        return (resp.text or "").strip()

    def generate(self, img, edit_prompt):
        try:
            resp = self.client.models.generate_content(
                model=self.image_model, contents=[edit_prompt, img])
        except Exception as e:
            raise _friendly(e, self.image_model) from e
        parts = resp.candidates[0].content.parts if resp.candidates else []
        for p in parts:
            if getattr(p, "inline_data", None) and p.inline_data.data:
                return Image.open(io.BytesIO(p.inline_data.data)).convert("RGB")
        text = " ".join(getattr(p, "text", "") or "" for p in parts).strip()
        raise EditError(f"The model returned no image. {text[:300]}")


class OpenAIEditor(BaseEditor):
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise EditError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.text_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

    def _vision(self, img, system, user):
        b64 = base64.b64encode(_png_bytes(img)).decode()
        r = self.client.chat.completions.create(
            model=self.text_model, temperature=0.1,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": [
                          {"type": "text", "text": user},
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        return r.choices[0].message.content or ""

    def interpret(self, img, request):
        return _parse_json(self._vision(img, prompts.INTERPRETER_SYSTEM, f"User request: {request}"))

    def caption(self, img):
        return self._vision(img, "You write concise image captions.", prompts.CAPTION_PROMPT).strip()

    def generate(self, img, edit_prompt):
        r = self.client.images.edit(model=self.image_model,
                                    image=("image.png", _png_bytes(img), "image/png"),
                                    prompt=edit_prompt)
        return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")


class HuggingFaceEditor(BaseEditor):
    """FREE option: Hugging Face Inference (monthly free credits with a free token).
    Image edit via an open model; the planner/captions reuse Gemini's FREE text model if a
    GEMINI_API_KEY is set (otherwise the planner is skipped)."""
    name = "huggingface"

    def __init__(self):
        from huggingface_hub import InferenceClient
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EditError("HF_TOKEN is not set. Create a free token at huggingface.co/settings/tokens "
                            "(enable 'Make calls to Inference Providers').")
        self.model = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
        self.client = InferenceClient(provider=os.getenv("HF_PROVIDER", "auto"), api_key=token)
        self._text = None
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                self._text = GeminiEditor()
            except Exception:
                self._text = None

    def interpret(self, img, request):
        return self._text.interpret(img, request) if self._text else None

    def caption(self, img):
        return self._text.caption(img) if self._text else ""

    def generate(self, img, edit_prompt):
        # Keep uploads small: faster, and friendlier to free-tier limits.
        small = img.copy()
        small.thumbnail((1024, 1024))
        try:
            out = self.client.image_to_image(_png_bytes(small), prompt=edit_prompt, model=self.model)
        except Exception as e:
            raise _friendly(e, self.model) from e
        if isinstance(out, (bytes, bytearray)):
            out = Image.open(io.BytesIO(out))
        return out.convert("RGB")


def _find_image(obj):
    """Dig a PIL-loadable image path out of whatever a Gradio Space returns."""
    if isinstance(obj, str):
        if os.path.isfile(obj) and obj.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return obj
        return None
    if isinstance(obj, dict):
        for k in ("path", "image", "url", "value"):
            if k in obj and (r := _find_image(obj[k])):
                return r
        for v in obj.values():
            if r := _find_image(v):
                return r
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if r := _find_image(v):
                return r
    return None


class HFSpaceEditor(BaseEditor):
    """FREE option 2: call a public Hugging Face Space (Gradio) as an API.
    Uses the Space's free ZeroGPU quota (a few minutes/day with a free token) instead of
    Inference-Provider credits. Works with many editing Spaces: it reads the Space's API
    schema, sends the image + prompt, and leaves every other parameter at its default."""
    name = "hfspace"

    def __init__(self):
        from gradio_client import Client
        self.space = _env("HF_SPACE", "Qwen/Qwen-Image-Edit-2509")
        self.api_name = _env("HF_SPACE_API")
        try:
            self.client = Client(self.space, token=_env("HF_TOKEN") or None, verbose=False)
        except Exception as e:
            raise EditError(f"Could not open Space '{self.space}': {str(e)[:200]}. "
                            "It may be asleep or removed - set HF_SPACE to another editing Space.")
        self._text = None
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                self._text = GeminiEditor()
            except Exception:
                self._text = None

    def interpret(self, img, request):
        return self._text.interpret(img, request) if self._text else None

    def caption(self, img):
        return self._text.caption(img) if self._text else ""

    def _endpoint(self):
        info = self.client.view_api(return_format="dict", print_info=False)
        eps = info.get("named_endpoints", {})
        if self.api_name and not self.api_name.startswith("/"):
            self.api_name = "/" + self.api_name
        if self.api_name:
            return self.api_name, eps.get(self.api_name, {})
        for name, ep in eps.items():  # first endpoint that takes an image and a prompt
            labels = " ".join(str(p.get("label", "")) + str(p.get("parameter_name", ""))
                              + str(p.get("component", "")) for p in ep.get("parameters", [])).lower()
            if ("image" in labels or "gallery" in labels) and ("prompt" in labels or "instruction" in labels):
                return name, ep
        raise EditError(f"No image-edit endpoint found on {self.space}. Set HF_SPACE_API.")

    def generate(self, img, edit_prompt):
        import tempfile
        from gradio_client import handle_file

        small = img.copy()
        small.thumbnail((768, 768))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            small.save(f.name)

        try:
            result = self.client.predict(
                input_image=handle_file(f.name),
                prompt=edit_prompt,
                seed=0,
                randomize_seed=True,
                guidance_scale=2.5,
                steps=20,
                api_name="/infer",
            )
        except Exception as e:
            raise _friendly(e, self.space) from e

        path = _find_image(result)
        if not path:
            raise EditError(f"{self.space} returned no image (got: {str(result)[:150]}).")
        return Image.open(path).convert("RGB")


class MockEditor(BaseEditor):
    """No API key needed: applies simple PIL filters so the UI/versioning can be demoed offline."""
    name = "mock"

    def caption(self, img):
        return f"Mock caption for a {img.width}x{img.height} image."

    def generate(self, img, edit_prompt):
        p = edit_prompt.lower()
        if "black and white" in p or "grayscale" in p:
            return ImageOps.grayscale(img).convert("RGB")
        if "warm" in p:
            r, g, b = img.split()
            return Image.merge("RGB", (r.point(lambda v: min(255, v + 25)), g, b.point(lambda v: v * 0.85)))
        if "cool" in p or "blue" in p:
            r, g, b = img.split()
            return Image.merge("RGB", (r.point(lambda v: v * 0.85), g, b.point(lambda v: min(255, v + 25))))
        if "sharp" in p or "enhance" in p:
            return ImageEnhance.Contrast(img.filter(ImageFilter.SHARPEN)).enhance(1.15)
        if "blur" in p:
            return img.filter(ImageFilter.GaussianBlur(4))
        return ImageEnhance.Color(img).enhance(1.4)  # generic visible change


def get_editor(provider: str | None = None) -> BaseEditor:
    provider = (provider or os.getenv("PROVIDER", "mock")).lower()
    if provider == "gemini":
        return GeminiEditor()
    if provider in ("hfspace", "space"):
        return HFSpaceEditor()
    if provider in ("huggingface", "hf"):
        return HuggingFaceEditor()
    if provider == "openai":
        return OpenAIEditor()
    if provider == "mock":
        return MockEditor()
    raise EditError(f"Unknown provider '{provider}'")
