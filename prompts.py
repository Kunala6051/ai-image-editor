"""Prompt engineering templates.

Two stages:
  1. INTERPRETER (text/vision LLM): turns a casual user request into a precise,
     structured edit spec (JSON). Optional, but makes the edit model far more
     reliable and lets us reject/clarify unsafe or impossible requests.
  2. EDIT_TEMPLATE (image model): wraps the structured spec in strict
     "change only this, keep everything else" guard-rails.
"""

INTERPRETER_SYSTEM = """You are the instruction-planner for an AI photo editor.
You receive an image and a casual user request. Produce ONE JSON object, no prose,
no markdown fences, with exactly these keys:
{
  "operation": one of ["remove_object","add_object","replace_background","remove_background",
                       "recolor","style_transfer","enhance","retouch","crop_reframe","other"],
  "target": short description of WHAT in the image is affected (use position/colour/size
            to disambiguate, e.g. "the person on the left wearing a red jacket"),
  "instruction": one clear imperative sentence an image-editing model can follow,
  "preserve": list of things that must NOT change (identity of people, composition, lighting...),
  "feasible": true/false,
  "reason_if_infeasible": string or ""
}
Rules: be literal, do not invent extra edits, resolve vague words ("it", "that") using
the image, and set feasible=false if the target does not exist in the image."""

EDIT_TEMPLATE = """You are editing the provided photo. Apply exactly ONE edit.

EDIT: {instruction}
TARGET: {target}
KEEP UNCHANGED: {preserve}

Rules:
- Change only what the edit requires. Keep the same framing, aspect ratio, resolution,
  camera angle, lighting direction and every untouched region pixel-identical in look.
- Any newly generated or filled-in area must match the surrounding lighting, texture,
  grain, shadows and perspective so the result looks like an unedited photograph.
- Do not add text, watermarks, borders or logos.
- Return the edited image only."""

# Fallback when the interpreter is skipped/unavailable
SIMPLE_TEMPLATE = """Edit this photo: {request}

Change only what is asked. Keep everything else, including framing, aspect ratio, people's
identity and lighting, exactly as is. Make new areas blend seamlessly. Return only the image."""

CAPTION_PROMPT = ("Describe this image in 2 sentences for a search index: main subjects, "
                  "setting, colours, mood. No preamble.")

# One-click presets -> pre-written natural-language requests
PRESETS = {
    "Remove background": "Remove the background and replace it with a clean plain white background, keeping the main subject untouched.",
    "Remove all objects": "Remove all small distracting objects and clutter, keeping only the main subject and the natural scene.",
    "Remove people": "Remove all people from the image and realistically fill in the background where they stood.",
    "Enhance quality": "Enhance the photo: improve sharpness, exposure and white balance subtly without changing content.",
    "Black & white": "Convert the photo to a high-quality black and white image with rich contrast.",
    "Warm tones": "Shift the colour grading to warm golden-hour tones without changing any content.",
    "Cool tones": "Shift the colour grading to cool blue tones without changing any content.",
    "Vintage film": "Give the photo a vintage 35mm film look: faded blacks, soft grain, slightly muted colours.",
}


def build_edit_prompt(spec: dict | None, request: str) -> str:
    """Spec from the interpreter -> strict edit prompt. Falls back to SIMPLE_TEMPLATE."""
    if not spec:
        return SIMPLE_TEMPLATE.format(request=request)
    preserve = ", ".join(spec.get("preserve") or ["everything not mentioned in the edit"])
    return EDIT_TEMPLATE.format(instruction=spec.get("instruction") or request,
                                target=spec.get("target") or "as described in the edit",
                                preserve=preserve)
