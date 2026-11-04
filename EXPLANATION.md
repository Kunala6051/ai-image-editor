# How API selection and prompts work

## 1. Division of labour between models
An image-generation model is good at *producing* pixels but poor at *understanding* a vague request.
A chat/vision model is the opposite. The app uses each for what it does best:

| Job | Model type | Used for |
|---|---|---|
| Interpret the request, caption images | Vision LLM (Gemini 3.8 Flash text model, free tier) | Turn "get rid of him" into a precise spec; write search captions later |
| Produce the edited image | Image-editing model (FLUX.1 Kontext via a Hugging Face Space) | Apply exactly one edit to the photo |

All providers implement the same three methods (`interpret`, `caption`, `generate`), so switching is a
dropdown, and a failure in one stage never needs code changes in the others.

## 2. Why these providers (constraints I hit)
1. **Gemini image model (`gemini-2.5-flash-image`)** was my first choice, but the free tier has no quota for
   image output (HTTP 429, `limit: 0`). It works on a paid plan.
2. **Hugging Face Inference Providers** worked in principle, but the free monthly credits ran out after very few requests.
3. **Qwen-Image-Edit Space** failed because it requests 600 s of GPU per call, more than a free account may request.
4. **FLUX.1 Kontext Space** requests a smaller GPU slot and works with a free token, so it is the default.
The provider interface made each switch a configuration change.

## 3. Prompt design
**Stage 1 - interpreter (system prompt in `prompts.py`).** The vision LLM sees the image and the user request and
returns JSON: `operation`, `target` (disambiguated by position/colour), `instruction`, `preserve`, `feasible`.
- Forces literal interpretation and forbids extra edits
- `feasible=false` stops impossible requests (target not in image) before spending GPU time
- If this stage fails or no key is set, the app falls back to a simple template; editing is never blocked

**Stage 2 - edit template.** The spec is slotted into a strict template that says: apply exactly one edit, keep
framing, lighting and untouched regions, blend new areas with surrounding texture and shadows, add no text or
watermarks. Edit models drift without these guard-rails.

**Presets.** The one-click buttons are pre-written natural-language requests that go through the same pipeline.

The UI panel "How was my request interpreted?" shows the spec and final prompt for every edit, which makes
prompt behaviour inspectable.

## 4. Version history
- Folder per image: `v0.png` (original, never overwritten), `v1.png`, ..., `thumbs/`, and `meta.json`
- Each version records: id, parent id, prompt, kind (original / edit / revert), provider, timestamp, size
- Editing from an older version creates a branch via the parent id
- Revert copies an old version forward as a new newest version, so no history is lost

## 5. Known limitations
Output quality depends on the free model; vague prompts and tiny targets can fail or alter other regions. Free GPU
quotas limit how many edits can be made per day.