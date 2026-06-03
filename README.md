# MangaTranslator

A local, detailed manga page translator. Upload a page → it detects every speech
bubble / text box, translates the Japanese with Claude, removes the original text,
and types the translation back **inside the same bubble** (auto-sized so it never
spills out). Optionally, clean up a rough sketch into a crisp "manga scan" first
using an OpenAI or Gemini image model.

## Features

- **Precise balloon detection** — a trained **YOLOv8 speech-balloon segmentation
  model** (GPU) outputs an exact pixel mask per balloon, so text removal and
  containment follow the *real* balloon shape. No GPU? It automatically falls back
  to a built-in enclosure-based OpenCV detector — nothing breaks.
- **Complete text removal** — the entire balloon interior is wiped (no Japanese
  bleed-through), while the inked outline and surrounding artwork stay untouched.
- **In-bubble typesetting** — a Pillow renderer fits the translation into the
  largest rectangle inscribed in the true balloon shape (auto-sized, wrapped,
  centered) so text never spills out.
- **SFX preserved** — sound effects / onomatopoeia and free art text are detected
  and left alone; only enclosed dialogue balloons are translated.
- **Translation** — Claude or Gemini Vision reads the page (original + numbered
  overlay) and returns a translation per region. A *Smart Detection* mode lets the
  model locate regions itself for tricky layouts.
- **Edit & reject** — review every balloon, edit text inline, or hit ✕ to skip one
  (keeps the original), then re-render instantly from the pristine base image.
- **Bulk mode** — upload a whole chapter, auto-translate all pages, reorder, and
  download a ZIP in your chosen order.
- **Rough → Manga Scan (AI cleanup)** — send a rough/pencil page to
  **OpenAI `gpt-image-1`** or **Gemini `gemini-2.5-flash-image`** with an editable
  prompt, then translate the cleaned result. Provider keys are saved in your browser.

## Run it

```bash
pip install -r requirements.txt        # core app (CPU; CV detector)
./setup_gpu.sh                         # optional: GPU balloon model (recommended)
python app.py
# open http://localhost:8000
```

`setup_gpu.sh` installs `torch` + `ultralytics` and pre-downloads the segmentation
model. After it runs, the Detect step shows **"segmentation model (GPU)"**. Without
it, you'll see **"CV detector"** and the app still works.

Paste your **Claude or Gemini API key** (for translation). For the optional
*Rough → Manga Scan* step, open that panel and add an **OpenAI or Gemini** key.

Override the model if you like:

```bash
export BUBBLE_MODEL_REPO="kitsumed/yolov8m_seg-speech-bubble"   # HF repo (default)
export BUBBLE_MODEL_PATH="/path/to/your/weights.pt"            # or a local .pt
```

## Fonts

The lettering font is auto-detected from `fonts/`. Bundled: **Bangers** and
**Comic Neue**. For the classic manga look, drop **Anime Ace** (`AnimeAce.ttf`,
free for personal use from dafont.com) into `fonts/` — or use the upload button in
the UI — and pick it from the Font dropdown.

## How it works

```
upload ─▶ [optional] Rough→Scan (OpenAI/Gemini) ─▶ segment balloons (GPU model
or CV fallback) ─▶ translate (Claude/Gemini) ─▶ wipe balloon interior ─▶
fit translation into the true balloon shape ─▶ review/edit ─▶ download / ZIP
```

The detector returns a precise interior **mask** per balloon. The compositor wipes
that exact region white (or black for dark balloons) and the renderer fits the
translation into the largest rectangle inscribed in the mask — which is why text
stays contained even in round or irregular balloons.

## Notes

- `gpt-image-1` may require OpenAI organization verification.
- Image-generation models can alter small text; for translation accuracy, prefer
  enhancing first and letting the bubble pipeline handle the text.
