# MangaTranslator — working notes for Claude Code

Local manga translator (Japanese → English): a FastAPI app (`app.py`) serving
a vanilla-JS page (`templates/index.html`, `static/js/app.js`,
`static/css/style.css`); the work happens in `core/`. The owner's team
scanlates One Piece and measures the output against TCB's releases.

## Run it

    python3 app.py                         # http://localhost:8000 (PORT=8001 to change)
    python3 app.py 2>&1 | tee output/app.log   # keep the log to read later
    curl -s localhost:8000/api/health | python3 -m json.tool   # what's loaded, build

The startup banner (`[pipeline] ===== component stack =====`) says which parts
run on the GPU. "text-pixel seg : CPU — slow!" → `./setup_gpu.sh --fix-onnx`.

Stop the app by its port (`fuser -k 8000/tcp`), not with `pkill -f "python app.py"`
— a pattern like that also matches the shell running the command.

## Translate pages the way the browser does

    python3 tools/run_page.py path/to/page.jpg --series "One Piece"
    python3 tools/run_page.py path/to/chapter_folder/ --series "One Piece" --set max_quality=true

Each page leaves in `output/runs/<time>/`: `<page>.out.*` (finished),
`<page>.raw.*` (original), `<page>.debug.txt` (every region: type,
in-balloon?, placed?, box, Japanese → English) and `<page>.status.json`. Any
`/api/translate` form field can be set with `--set name=value` (see its
parameters in `app.py`).

The owner's test pages (Windows folders, seen from Ubuntu): Japanese raws in
`/mnt/c/Users/Admin/MT/raws/<chapter>/`; TCB's real English releases, one
folder per chapter (1173-1193, 1195), in `/mnt/c/Users/Admin/MT/tcb_real/<chapter>/`
— the standard to match. `/mnt/c/Users/Admin/MT/tcb/1194/` is ANOTHER group's
release of 1194 (not TCB): useful to see what each balloon says, not how to letter it.

The offline engine self-tests its model on load: fugumt-ja-en fails here
(garbage on textbook sentences) and OPUS-MT ja-en is used — the log says
`[local-mt] ... failed its self-test` once per start. That is expected.

**Testing spends no API credits.** The owner's rule: test runs are free.

- Default engine is the offline one (`--provider local`, needs
  `python3 setup_models.py --offline-translate --langs japanese` once).
  Detection, reading, erasing and lettering are all real; only the English
  is rough, and the Gemini-only passes (AI free-text finder, picture checks)
  don't run. Use it for erase / layout / sizing / balloon work.
- To exercise the Gemini code path (reply matching, free-text passes), use the
  stub — fake lines, no key, no cost:

      python3 tests/ui/stub_gemini.py 8132 &
      GEMINI_BASE_URL=http://127.0.0.1:8132 python3 app.py
      python3 tools/run_page.py page.jpg --provider gemini

- Real Gemini/Claude is refused unless `--spend DOLLARS` is given. Don't
  pass it unless the owner asks for a paid run; then it uses the cheapest
  model (Flash-Lite), prints each page's cost, and stops before the cap.
  The key comes from `.env` — never put it on the command line.

## Tests

Plain scripts, no pytest: `for f in tests/test_*.py; do python3 $f || echo FAIL $f; done`.
Each prints `ALL CHECKS PASSED`. Browser tests are in `tests/ui/` (Chromium via
Playwright, against a running app + the stubs; each file's docstring says how).

## Where things happen

- `core/pipeline.py` — the page flow: balloon detection (`_standard_detect`),
  manga-ocr reads, the translate call (`_translate_regions`; replies are paired
  to bubbles by the Japanese they echo — `_match_replies`), free-text passes
  (`_free_text_llm`, `_free_text_seg`, `_free_text_gauntlet`,
  `_verify_text_region`), `tidy_free_text`.
- `core/compositor.py` — erasing and lettering: balloon checks
  (`_is_real_balloon`, `_outline_coverage`, `_resolve_bubble`), erase
  (`_inpaint_text`, `_fill_mask`, outlined letters `_outlined_glyphs` /
  `_erase_outlined`), sizing (`_glyph_px`, `_size_cap`, `_grow_for_presence`),
  joined balloons (`_lobes`, `_split_for_lobes`), credit placement.
- `core/renderer.py` — fitting and drawing text (balloon-shaped layout,
  outlines, letter spacing).
- `core/prompts.py` — every prompt; `core/translator.py` — Claude / Gemini calls.
- `core/series.py` + `core/series_data/one_piece.txt` — the One Piece preset
  (TCB's conventions, glossary, phrasebook of sounds). Add words there.
- `core/text_seg.py` — comic-text-detector (text pixels and blocks).

## How to fix a problem

1. Reproduce it on the page with `tools/run_page.py`; read `<page>.debug.txt`
   and the app log (`[pipeline]` / `[compositor]` lines say why a region was
   dropped, refused as a balloon, or moved), and look at the output image.
2. Find the cause in the code path above — fix causes, not the one page.
3. Add `tests/test_<thing>.py` that fails on the old code and passes on the fix
   (synthetic images or stubbed OCR/translator are fine).
4. Run all `tests/test_*.py`, then the page (and a few others) again.

## House rules

- The team's standard is TCB's release: sound effects drawn on the art stay
  untouched; balloon text fully erased; English sized and styled like the
  original lettering (outlined letters stay outlined); nothing lettered or
  stamped over other text or faces.
- Improvements to the UI are welcome, but don't change what features do
  without being asked.
- Work on the branch `claude/fervent-allen-i7euqk`; commit with clear messages.
- Never write the name of the code-hosting service or any account handle in
  project files, comments or commit messages.
