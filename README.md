# MangaTranslator

A local, detailed manga page translator. Upload a page → it detects every speech
bubble / text box, translates the Japanese with Claude, removes the original text,
and types the translation back **inside the same bubble** (auto-sized so it never
spills out). Optionally, clean up a rough sketch into a crisp "manga scan" first
using an OpenAI or Gemini image model.

## Features

- **Bubble detection** — OpenCV connected-component analysis finds white speech
  bubbles and text boxes, builds a precise text mask for each.
- **Translation** — Claude Vision reads the page (original + numbered overlay) and
  returns a translation per region. A *Smart Detection* mode lets Claude locate the
  text regions itself for tricky layouts.
- **Clean removal** — white-fill for white bubbles, Navier–Stokes inpainting elsewhere.
- **In-bubble typesetting** — Pillow renderer binary-searches the largest font size
  that wraps and fits inside each bubble, centered, so text stays contained.
- **Rough → Manga Scan (AI cleanup)** — send a rough/pencil page to
  **OpenAI `gpt-image-1`** or **Gemini `gemini-2.5-flash-image`** with an editable
  prompt, then translate the cleaned result. Provider keys are saved in your browser.
- **Polished local web UI** — drag & drop, live step progress, and a before/after
  comparison slider. Everything runs on `localhost`; your API keys never leave your machine.

## Run it

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

Paste your **Claude API key** (for translation). For the optional *Rough → Manga
Scan* step, open that panel and add an **OpenAI or Gemini** key.

## Fonts

The lettering font is auto-detected from `fonts/`. Bundled: **Bangers** and
**Comic Neue**. For the classic manga look, drop **Anime Ace** (`AnimeAce.ttf`,
free for personal use from dafont.com) into `fonts/` — or use the upload button in
the UI — and pick it from the Font dropdown.

## How it works

```
upload ─▶ [optional] Rough→Scan (OpenAI/Gemini) ─▶ detect bubbles ─▶
translate (Claude) ─▶ remove text ─▶ render translation in-bubble ─▶ download
```

## Notes

- `gpt-image-1` may require OpenAI organization verification.
- Image-generation models can alter small text; for translation accuracy, prefer
  enhancing first and letting the bubble pipeline handle the text.
